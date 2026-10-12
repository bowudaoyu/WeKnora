package chat

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestOllamaBuildChatRequest_QwenThinkingOff(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "hi"}}

	qwen := &OllamaChat{modelName: "qwen3:8b"}
	req := qwen.buildChatRequest(msgs, &ChatOptions{Thinking: ptrBool(true)}, true)
	require.NotNil(t, req.Think)
	assert.False(t, req.Think.Bool())

	req = qwen.buildChatRequest(msgs, nil, false)
	require.NotNil(t, req.Think)
	assert.False(t, req.Think.Bool())

	other := &OllamaChat{modelName: "llama3"}
	req = other.buildChatRequest(msgs, &ChatOptions{Thinking: ptrBool(true)}, true)
	require.NotNil(t, req.Think)
	assert.True(t, req.Think.Bool())
}
