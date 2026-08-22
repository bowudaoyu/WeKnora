package service

import (
	"strings"
	"testing"

	"github.com/Tencent/WeKnora/internal/types"
)

func TestNewScanOCRBackendFromEnv_DisabledByDefault(t *testing.T) {
	t.Setenv("SCAN_OCR_BASE_URL", "")

	b, err := newScanOCRBackendFromEnv()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if b != nil {
		t.Fatalf("expected nil backend when SCAN_OCR_BASE_URL is unset, got %+v", b)
	}
	// A nil backend must never claim a page — that is what keeps the original
	// VLM path intact for every deployment that has not opted in.
	if b.handles(types.ImageMultimodalPayload{ImageSourceType: "scanned_pdf"}) {
		t.Error("nil backend must not handle scanned pages")
	}
}

func TestNewScanOCRBackendFromEnv_Defaults(t *testing.T) {
	t.Setenv("SCAN_OCR_BASE_URL", "http://127.0.0.1:9800/v1")

	b, err := newScanOCRBackendFromEnv()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if b == nil {
		t.Fatal("expected backend to be enabled")
	}
	if b.modelName != defaultScanOCRModelName {
		t.Errorf("model name = %q, want %q", b.modelName, defaultScanOCRModelName)
	}
	if b.prompt != defaultScanOCRPrompt {
		t.Errorf("prompt should default to the published OvisOCR2 instruction")
	}
	if !b.skipCaption {
		t.Error("captioning should be skipped for scanned pages by default")
	}
}

func TestNewScanOCRBackendFromEnv_Overrides(t *testing.T) {
	t.Setenv("SCAN_OCR_BASE_URL", "http://127.0.0.1:9800/v1")
	t.Setenv("SCAN_OCR_MODEL_NAME", "Unlimited-OCR")
	t.Setenv("SCAN_OCR_PROMPT", "<image>document parsing.")
	t.Setenv("SCAN_OCR_SKIP_CAPTION", "false")

	b, err := newScanOCRBackendFromEnv()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if b.modelName != "Unlimited-OCR" {
		t.Errorf("model name = %q", b.modelName)
	}
	if b.prompt != "<image>document parsing." {
		t.Errorf("prompt = %q", b.prompt)
	}
	if b.skipCaption {
		t.Error("SCAN_OCR_SKIP_CAPTION=false should re-enable captioning")
	}
}

func TestScanOCRBackend_HandlesOnlyScannedPages(t *testing.T) {
	t.Setenv("SCAN_OCR_BASE_URL", "http://127.0.0.1:9800/v1")
	b, err := newScanOCRBackendFromEnv()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !b.handles(types.ImageMultimodalPayload{ImageSourceType: "scanned_pdf"}) {
		t.Error("scanned_pdf pages should be routed to the local backend")
	}
	// Ordinary illustrations must keep going to the KB's VLM: describing a
	// picture is exactly what a document-parsing model cannot do.
	for _, src := range []string{"", "embedded", "upload"} {
		if b.handles(types.ImageMultimodalPayload{ImageSourceType: src}) {
			t.Errorf("image_source_type=%q must not be routed to the OCR backend", src)
		}
	}
}

func TestCleanScanOCRText(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{
			name: "strips bbox figure placeholders",
			in:   "# 第一章\n\n<img src=\"images/bbox_10_20_300_400.jpg\" />\n\n正文内容。",
			want: "# 第一章\n\n\n\n正文内容。",
		},
		{
			name: "strips placeholders without the self-closing slash",
			in:   "前<img src=\"images/bbox_1_2_3_4.jpg\">后",
			want: "前后",
		},
		{
			name: "keeps HTML tables, which carry real content",
			in:   "<table><tr><td>甲</td></tr></table>",
			want: "<table><tr><td>甲</td></tr></table>",
		},
		{
			name: "keeps genuine image references to extracted files",
			in:   "![图版](images/plate_01.jpg)",
			want: "![图版](images/plate_01.jpg)",
		},
		{
			name: "trims surrounding whitespace",
			in:   "\n\n  正文  \n\n",
			want: "正文",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := cleanScanOCRText(tt.in); got != tt.want {
				t.Errorf("cleanScanOCRText() = %q, want %q", got, tt.want)
			}
		})
	}
}

// The cleaned output still passes through sanitizeOCRText, so the two must
// compose without the sanitizer throwing away a legitimate page.
func TestCleanScanOCRTextComposesWithSanitizer(t *testing.T) {
	raw := "<img src=\"images/bbox_0_0_1000_1000.jpg\" />\n\n# 卷一\n\n" +
		strings.Repeat("扫描书籍的正文段落。", 5)

	got := sanitizeOCRText(cleanScanOCRText(raw))
	if !strings.Contains(got, "# 卷一") {
		t.Errorf("heading was lost: %q", got)
	}
	if strings.Contains(got, "bbox_") {
		t.Errorf("bbox placeholder survived: %q", got)
	}
}

func TestEnvBoolOrDefault(t *testing.T) {
	tests := []struct {
		val  string
		def  bool
		want bool
	}{
		{"", true, true},
		{"", false, false},
		{"true", false, true},
		{"ON", false, true},
		{"1", false, true},
		{"false", true, false},
		{"no", true, false},
		{"off", true, false},
		// An unparseable value must fall back to the default rather than
		// silently flipping behaviour.
		{"maybe", true, true},
		{"maybe", false, false},
	}

	for _, tt := range tests {
		t.Setenv("SCAN_OCR_TEST_BOOL", tt.val)
		if got := envBoolOrDefault("SCAN_OCR_TEST_BOOL", tt.def); got != tt.want {
			t.Errorf("envBoolOrDefault(%q, %v) = %v, want %v", tt.val, tt.def, got, tt.want)
		}
	}
}

func TestScanOCRConfigured(t *testing.T) {
	t.Setenv("SCAN_OCR_BASE_URL", "")
	if scanOCRConfigured() {
		t.Error("unset SCAN_OCR_BASE_URL must report not configured")
	}
	t.Setenv("SCAN_OCR_BASE_URL", "   ")
	if scanOCRConfigured() {
		t.Error("blank SCAN_OCR_BASE_URL must report not configured")
	}
	t.Setenv("SCAN_OCR_BASE_URL", "http://127.0.0.1:9800/v1")
	if !scanOCRConfigured() {
		t.Error("set SCAN_OCR_BASE_URL must report configured")
	}
}
