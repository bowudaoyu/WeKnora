package chat

import (
	"context"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Tencent/WeKnora/internal/tracing/langfuse"
	"github.com/Tencent/WeKnora/internal/types"
)

func TestBuildLangfuseGenerationOutput(t *testing.T) {
	toolCalls := []types.LLMToolCall{{ID: "call_1", Type: "function"}}

	got := buildLangfuseGenerationOutput("", "", "tool_calls", toolCalls)
	want := map[string]interface{}{
		"content":       "",
		"tool_calls":    toolCalls,
		"finish_reason": "tool_calls",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("output without reasoning = %#v; want %#v", got, want)
	}

	got = buildLangfuseGenerationOutput("answer", "thinking", "stop", nil)
	want = map[string]interface{}{
		"content":           "answer",
		"tool_calls":        []types.LLMToolCall(nil),
		"finish_reason":     "stop",
		"reasoning_content": "thinking",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("output with reasoning = %#v; want %#v", got, want)
	}
}

func TestBuildLangfuseMessagesReasoningContent(t *testing.T) {
	msgs := buildLangfuseMessages([]Message{
		{Role: "assistant", ReasoningContent: "chain of thought", ToolCalls: []ToolCall{{ID: "tc1"}}},
	})
	if len(msgs) != 1 {
		t.Fatalf("len(messages) = %d; want 1", len(msgs))
	}
	if msgs[0]["reasoning_content"] != "chain of thought" {
		t.Fatalf("reasoning_content = %v; want chain of thought", msgs[0]["reasoning_content"])
	}
}

type fakeStreamGen struct {
	mu       sync.Mutex
	finishes int
	output   interface{}
	err      error
	done     chan struct{}
}

func newFakeStreamGen() *fakeStreamGen { return &fakeStreamGen{done: make(chan struct{}, 4)} }

func (f *fakeStreamGen) MarkCompletionStart(time.Time) {}

func (f *fakeStreamGen) Finish(output interface{}, _ *langfuse.TokenUsage, err error) {
	f.mu.Lock()
	f.finishes++
	f.output, f.err = output, err
	f.mu.Unlock()
	f.done <- struct{}{}
}

func waitFinish(t *testing.T, f *fakeStreamGen) {
	t.Helper()
	select {
	case <-f.done:
	case <-time.After(2 * time.Second):
		t.Fatal("generation was never finished")
	}
}

func TestRelayStreamRecordsErrorEvent(t *testing.T) {
	ch := make(chan types.StreamResponse, 3)
	ch <- types.StreamResponse{ResponseType: types.ResponseTypeAnswer, Content: "部分"}
	ch <- types.StreamResponse{ResponseType: types.ResponseTypeError, Content: "upstream 500"}
	close(ch)

	gen := newFakeStreamGen()
	var got []types.StreamResponse
	for r := range relayStreamToGeneration(context.Background(), ch, gen) {
		got = append(got, r)
	}
	waitFinish(t, gen)

	if len(got) != 2 {
		t.Fatalf("forwarded %d events; want 2", len(got))
	}
	if gen.err == nil || gen.err.Error() != "upstream 500" {
		t.Fatalf("finish err = %v; want upstream 500", gen.err)
	}
	if out := gen.output.(map[string]interface{}); out["content"] != "部分" {
		t.Fatalf("content = %v; want 部分", out["content"])
	}
}

func TestRelayStreamFinishesWhenCallerAbandons(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	ch := make(chan types.StreamResponse)
	gen := newFakeStreamGen()
	wrapped := relayStreamToGeneration(ctx, ch, gen)

	ch <- types.StreamResponse{ResponseType: types.ResponseTypeAnswer, Content: "a"}
	<-wrapped // caller reads one chunk, then goes away
	cancel()
	ch <- types.StreamResponse{ResponseType: types.ResponseTypeAnswer, Content: "b"}
	waitFinish(t, gen)
	if gen.err == nil || !strings.Contains(gen.err.Error(), "abandoned") {
		t.Fatalf("finish err = %v; want abandoned", gen.err)
	}

	// The producer must still be able to push the rest without blocking.
	sent := make(chan struct{})
	go func() {
		ch <- types.StreamResponse{ResponseType: types.ResponseTypeAnswer, Content: "c"}
		close(ch)
		close(sent)
	}()
	select {
	case <-sent:
	case <-time.After(2 * time.Second):
		t.Fatal("producer blocked after caller abandoned the stream")
	}
	time.Sleep(50 * time.Millisecond)
	gen.mu.Lock()
	defer gen.mu.Unlock()
	if gen.finishes != 1 {
		t.Fatalf("Finish called %d times; want 1", gen.finishes)
	}
}
