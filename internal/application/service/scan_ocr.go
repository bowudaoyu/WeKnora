package service

import (
	"context"
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"

	"github.com/Tencent/WeKnora/internal/logger"
	"github.com/Tencent/WeKnora/internal/models/vlm"
	"github.com/Tencent/WeKnora/internal/types"
)

// Scanned book pages are the single most expensive thing this service does: a
// 300-page scan produces 300 full-page images, and routing each of them at a
// general-purpose cloud VLM (twice — OCR plus caption) is what makes ingesting
// a library unaffordable. Image tokens, not text tokens, are the bill.
//
// A dedicated end-to-end document-parsing model solves both halves of that:
// it is small enough to self-host (OvisOCR2 is 0.8B, Apache-2.0, and tops
// OmniDocBench v1.6), and it is *better* at dense full-page transcription than
// a general VLM, which tends to silently drop lines and then paper over the
// gap with fluent invented text — the worst possible failure mode for a RAG
// index, because nothing errors and nothing looks wrong.
//
// This backend is opt-in and purely env-driven (SCAN_OCR_BASE_URL enables it),
// matching how the rest of WeKnora selects pluggable engines. When it is not
// configured, every scanned page keeps taking the original VLM path.

// defaultScanOCRPrompt is the instruction published with OvisOCR2
// (ATH-MaaS/OvisOCR2). Document-parsing models are trained against one exact
// prompt and degrade badly on a paraphrase, so this string is copied verbatim
// and is overridable via SCAN_OCR_PROMPT when pointing at a different model.
const defaultScanOCRPrompt = "Extract all readable content from the image in natural human reading order " +
	"and output the result as a single Markdown document. For charts or images, represent them using an " +
	"HTML image tag: <img src=\"images/bbox_{left}_{top}_{right}_{bottom}.jpg\" />, where left, top, right, " +
	"bottom are bounding box coordinates scaled to [0, 1000). Format formulas as LaTeX. Format tables as " +
	"HTML: <table>...</table>. Transcribe all other text as standard Markdown. Preserve the original text " +
	"without translation or paraphrasing."

const (
	defaultScanOCRModelName = "OvisOCR2"
	// Document-parsing models emit a whole page of Markdown in one shot; the
	// generic VLM ceiling (5000) truncates dense pages mid-table. 16384 is the
	// value OvisOCR2 ships with.
	defaultScanOCRMaxTokens = 16384
)

// scanOCRBBoxImage matches the bounding-box image placeholders OvisOCR2 emits
// for figures. They point at files that were never extracted, so in a RAG
// chunk they are pure noise — strip them rather than indexing dead paths.
var scanOCRBBoxImage = regexp.MustCompile(`(?i)<img[^>]*\bsrc\s*=\s*"images/bbox_[^"]*"[^>]*>`)

// scanOCRBackend is a self-hosted, OpenAI-compatible document-parsing model
// used *only* for pages docreader classified as scanned (ImageSourceType
// "scanned_pdf"). Ordinary illustrations still go to the knowledge base's
// configured VLM, which is the right tool for "describe this picture".
type scanOCRBackend struct {
	model     vlm.VLM
	modelName string
	prompt    string
	// skipCaption drops the second (caption) VLM call for scanned pages.
	// "A page of scanned text" is a useless caption that costs a full extra
	// image round-trip and pollutes the index with a near-duplicate chunk.
	skipCaption bool
}

// newScanOCRBackendFromEnv builds the backend from the environment, returning
// (nil, nil) when SCAN_OCR_BASE_URL is unset — i.e. the feature is off and the
// original behaviour is preserved verbatim.
func newScanOCRBackendFromEnv() (*scanOCRBackend, error) {
	baseURL := strings.TrimSpace(os.Getenv("SCAN_OCR_BASE_URL"))
	if baseURL == "" {
		return nil, nil
	}

	modelName := envOrDefault("SCAN_OCR_MODEL_NAME", defaultScanOCRModelName)
	prompt := envOrDefault("SCAN_OCR_PROMPT", defaultScanOCRPrompt)

	// Self-hosted vLLM/SGLang servers started without --api-key accept any
	// bearer token, but go-openai must send *something* for gateways that
	// insist on the header being present.
	apiKey := envOrDefault("SCAN_OCR_API_KEY", "EMPTY")

	maxTokens := defaultScanOCRMaxTokens
	if v := strings.TrimSpace(os.Getenv("SCAN_OCR_MAX_TOKENS")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			maxTokens = n
		}
	}

	// Transcription must be deterministic: any sampling at all invites the
	// model to invent plausible characters on a smudged scan.
	temperature := float32(0)
	if v := strings.TrimSpace(os.Getenv("SCAN_OCR_TEMPERATURE")); v != "" {
		if f, err := strconv.ParseFloat(v, 32); err == nil && f >= 0 {
			temperature = float32(f)
		}
	}

	model, err := vlm.NewVLM(&vlm.Config{
		Source:        types.ModelSourceRemote,
		InterfaceType: "openai",
		BaseURL:       baseURL,
		ModelName:     modelName,
		APIKey:        apiKey,
		ModelID:       "scan_ocr:" + modelName,
		MaxTokens:     maxTokens,
		Temperature:   &temperature,
	}, nil)
	if err != nil {
		return nil, fmt.Errorf("build scanned-page OCR backend (%s): %w", baseURL, err)
	}

	return &scanOCRBackend{
		model:       model,
		modelName:   modelName,
		prompt:      prompt,
		skipCaption: envBoolOrDefault("SCAN_OCR_SKIP_CAPTION", true),
	}, nil
}

// Recognize transcribes one scanned page to Markdown.
func (b *scanOCRBackend) Recognize(ctx context.Context, imgBytes []byte) (string, error) {
	out, err := b.model.Predict(ctx, [][]byte{imgBytes}, b.prompt)
	if err != nil {
		return "", err
	}
	return cleanScanOCRText(out), nil
}

// handles reports whether this backend should take over the page.
func (b *scanOCRBackend) handles(payload types.ImageMultimodalPayload) bool {
	return b != nil && payload.ImageSourceType == "scanned_pdf"
}

// cleanScanOCRText removes document-parsing markup that carries no meaning
// once the page is a text chunk. sanitizeOCRText still runs afterwards to
// apply the shared HTML/empty-reply handling.
func cleanScanOCRText(raw string) string {
	text := scanOCRBBoxImage.ReplaceAllString(raw, "")
	return strings.TrimSpace(text)
}

func envOrDefault(key, def string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return def
}

func envBoolOrDefault(key string, def bool) bool {
	v := strings.ToLower(strings.TrimSpace(os.Getenv(key)))
	if v == "" {
		return def
	}
	switch v {
	case "1", "true", "yes", "y", "on":
		return true
	case "0", "false", "no", "n", "off":
		return false
	}
	logger.Warnf(context.Background(), "unrecognised boolean for %s: %q, using %v", key, v, def)
	return def
}
