package chat

import (
	"strings"

	"github.com/sashabaranov/go-openai"
)

// ExtraConfigThinkingControl is the model parameters.extra_config key for
// selecting how ChatOptions.Thinking is translated to provider HTTP fields.
// The accepted values mirror the strings the frontend writes (see
// ModelEditorDialog.vue): "none", "enable_thinking", "thinking_type",
// "chat_template_kwargs".
const ExtraConfigThinkingControl = "thinking_control"

// Wire-format request bodies used by providers that express extended-thinking
// through a non-standard top-level field. They embed the standard OpenAI
// request so all other fields are marshalled unchanged.

// QwenChatCompletionRequest adds Aliyun Qwen's `enable_thinking` and
// `enable_search` booleans.
type QwenChatCompletionRequest struct {
	openai.ChatCompletionRequest
	EnableThinking *bool `json:"enable_thinking,omitempty"`
	EnableSearch   *bool `json:"enable_search,omitempty"`
}

// ThinkingConfig is the `{ "type": "enabled"|"disabled" }` block used by
// LKEAP / Volcengine style providers.
type ThinkingConfig struct {
	Type string `json:"type"`
}

// ThinkingChatCompletionRequest adds the `thinking` object for providers that
// use the `{ "thinking": { "type": ... } }` wire format.
type ThinkingChatCompletionRequest struct {
	openai.ChatCompletionRequest
	Thinking *ThinkingConfig `json:"thinking,omitempty"`
}

// ThinkingStrategy encodes how ChatOptions.Thinking is mapped onto a provider's
// HTTP request. Apply returns (customBody, useRawHTTP):
//   - (nil, false) means "send the standard OpenAI request unchanged" (the
//     caller keeps using the SDK path).
//   - a non-nil customBody must be sent verbatim over raw HTTP because it
//     carries fields the OpenAI SDK would strip.
//
// When opts.Thinking is nil most strategies emit nothing, deferring to the
// model's own default; the exception is qwenNoThinking (Aliyun Qwen), which
// always pins thinking and search off.
type ThinkingStrategy interface {
	Apply(req *openai.ChatCompletionRequest, opts *ChatOptions, isStream bool) (customBody any, useRawHTTP bool)
}

// noThinking sends no thinking-related fields at all.
type noThinking struct{}

func (noThinking) Apply(*openai.ChatCompletionRequest, *ChatOptions, bool) (any, bool) {
	return nil, false
}

// enableThinking encodes thinking via the top-level `enable_thinking` boolean
// (extra_config override). Emits nothing when opts.Thinking is unset.
type enableThinking struct{}

func (enableThinking) Apply(req *openai.ChatCompletionRequest, opts *ChatOptions, _ bool) (any, bool) {
	if opts == nil || opts.Thinking == nil {
		return nil, false
	}
	qwenReq := QwenChatCompletionRequest{ChatCompletionRequest: *req}
	qwenReq.EnableThinking = opts.Thinking
	return qwenReq, true
}

// qwenNoThinking pins Aliyun DashScope Qwen requests to enable_thinking=false
// and enable_search=false on every call, ignoring opts.Thinking: 项目约定所有
// Qwen 调用一律关闭思考与联网搜索。
type qwenNoThinking struct{}

func (qwenNoThinking) Apply(req *openai.ChatCompletionRequest, _ *ChatOptions, _ bool) (any, bool) {
	off := false
	qwenReq := QwenChatCompletionRequest{ChatCompletionRequest: *req}
	qwenReq.EnableThinking = &off
	qwenReq.EnableSearch = &off
	return qwenReq, true
}

// thinkingTypeField encodes thinking via the `{ "thinking": { "type": ... } }`
// object (LKEAP / Volcengine). Emits nothing when opts.Thinking is unset.
type thinkingTypeField struct{}

func (thinkingTypeField) Apply(req *openai.ChatCompletionRequest, opts *ChatOptions, _ bool) (any, bool) {
	if opts == nil || opts.Thinking == nil {
		return nil, false
	}
	r := ThinkingChatCompletionRequest{ChatCompletionRequest: *req}
	thinkingType := "disabled"
	if *opts.Thinking {
		thinkingType = "enabled"
	}
	r.Thinking = &ThinkingConfig{Type: thinkingType}
	return r, true
}

// chatTemplateKwargs encodes thinking via the standard request's
// `chat_template_kwargs.enable_thinking` (vLLM / NVIDIA / generic local
// deployments). Emits nothing when opts.Thinking is unset.
type chatTemplateKwargs struct{}

func (chatTemplateKwargs) Apply(req *openai.ChatCompletionRequest, opts *ChatOptions, _ bool) (any, bool) {
	if opts == nil || opts.Thinking == nil {
		return nil, false
	}
	req.ChatTemplateKwargs = map[string]interface{}{
		"enable_thinking": *opts.Thinking,
	}
	return req, true
}

// parseThinkingOverride reads extra_config.thinking_control and returns the
// strategy it selects, or nil when unset (the provider adapter's default
// strategy then applies). An unrecognized non-empty value falls back to
// chat_template_kwargs, preserving the legacy default-mode behavior.
func parseThinkingOverride(extraConfig map[string]string) ThinkingStrategy {
	if extraConfig == nil {
		return nil
	}
	switch strings.ToLower(strings.TrimSpace(extraConfig[ExtraConfigThinkingControl])) {
	case "":
		return nil
	case "none":
		return noThinking{}
	case "enable_thinking":
		return enableThinking{}
	case "thinking_type":
		return thinkingTypeField{}
	default:
		// "chat_template_kwargs" and any unknown non-empty value.
		return chatTemplateKwargs{}
	}
}
