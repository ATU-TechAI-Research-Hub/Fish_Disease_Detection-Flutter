import 'dart:async';

import 'package:flutter/material.dart';

import '../models/assistant_message.dart';
import '../services/assistant_controller.dart';
import '../theme/app_theme.dart';
import 'assistant_message_bubble.dart';

class AssistantPanel extends StatefulWidget {
  const AssistantPanel({
    required this.controller,
    super.key,
  });

  final AssistantController controller;

  @override
  State<AssistantPanel> createState() => _AssistantPanelState();
}

class _AssistantPanelState extends State<AssistantPanel> {
  final TextEditingController _input = TextEditingController();
  final ScrollController _scroll = ScrollController();
  final FocusNode _focusNode = FocusNode();
  String _lastMessageSignature = '';

  @override
  void initState() {
    super.initState();
    _scheduleScroll();
  }

  @override
  void didUpdateWidget(covariant AssistantPanel oldWidget) {
    super.didUpdateWidget(oldWidget);
    final messages = widget.controller.messages;
    final signature = messages.isEmpty
        ? ''
        : '${messages.length}:${messages.last.content.length}:'
            '${messages.last.isStreaming}';
    if (signature != _lastMessageSignature) {
      _lastMessageSignature = signature;
      _scheduleScroll();
    }
  }

  void _scheduleScroll() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted || !_scroll.hasClients) return;
      _scroll.animateTo(
        _scroll.position.maxScrollExtent,
        duration: const Duration(milliseconds: 220),
        curve: Curves.easeOut,
      );
    });
  }

  Future<void> _send() async {
    final message = _input.text.trim();
    if (message.isEmpty || widget.controller.isStreaming) return;
    _input.clear();
    await widget.controller.sendMessage(message);
    if (mounted) _focusNode.requestFocus();
  }

  Future<void> _confirmClear() async {
    if (widget.controller.messages.isEmpty) return;
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        title: const Text('Clear conversation?'),
        content: const Text(
          'This removes the local assistant messages for this session.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(dialogContext, false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(dialogContext, true),
            child: const Text('Clear'),
          ),
        ],
      ),
    );
    if (confirmed == true) {
      await widget.controller.clearConversation();
    }
  }

  @override
  void dispose() {
    _input.dispose();
    _scroll.dispose();
    _focusNode.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final controller = widget.controller;
    final messages = controller.messages;
    final lastAssistantIndex = messages.lastIndexWhere(
      (message) => message.role == AssistantRole.assistant,
    );

    return Material(
      color: context.surfaceCard,
      elevation: 18,
      shadowColor: Colors.black.withValues(alpha: 0.35),
      borderRadius: BorderRadius.circular(24),
      clipBehavior: Clip.antiAlias,
      child: Column(
        children: [
          _PanelHeader(
            controller: controller,
            onClear: _confirmClear,
          ),
          if (controller.currentPrediction != null)
            _PredictionContextBanner(controller: controller),
          Divider(height: 1, color: context.subtleBorder),
          Expanded(
            child: controller.isLoadingHistory && messages.isEmpty
                ? const Center(child: CircularProgressIndicator())
                : messages.isEmpty
                    ? _WelcomeView(onSuggestion: (text) {
                        _input.text = text;
                        unawaited(_send());
                      })
                    : ListView.builder(
                        controller: _scroll,
                        padding: const EdgeInsets.fromLTRB(14, 16, 10, 8),
                        itemCount: messages.length,
                        itemBuilder: (context, index) {
                          final message = messages[index];
                          return AssistantMessageBubble(
                            key: ValueKey(message.id),
                            message: message,
                            // Errors keep the regenerate button so a failed
                            // generation can be retried without retyping.
                            onRegenerate: index == lastAssistantIndex &&
                                    !controller.isStreaming
                                ? () => unawaited(
                                      controller.regenerateLast(),
                                    )
                                : null,
                          );
                        },
                      ),
          ),
          Divider(height: 1, color: context.subtleBorder),
          _Composer(
            input: _input,
            focusNode: _focusNode,
            isStreaming: controller.isStreaming,
            onSend: _send,
          ),
        ],
      ),
    );
  }
}

class _PanelHeader extends StatelessWidget {
  const _PanelHeader({
    required this.controller,
    required this.onClear,
  });

  final AssistantController controller;
  final VoidCallback onClear;

  @override
  Widget build(BuildContext context) {
    final availableModels =
        controller.models.where((model) => model.available).toList();
    return Container(
      padding: const EdgeInsets.fromLTRB(14, 10, 8, 10),
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          colors: [AppColors.deepOcean, AppColors.ocean],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
      ),
      child: Row(
        children: [
          Container(
            width: 34,
            height: 34,
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.12),
              borderRadius: BorderRadius.circular(11),
            ),
            child: const Icon(
              Icons.auto_awesome_rounded,
              color: AppColors.seafoam,
              size: 20,
            ),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Aquaculture Assistant',
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 14,
                    fontWeight: FontWeight.w800,
                  ),
                ),
                const SizedBox(height: 2),
                Row(
                  children: [
                    Container(
                      width: 7,
                      height: 7,
                      decoration: BoxDecoration(
                        color: controller.backendAvailable
                            ? AppColors.seafoam
                            : AppColors.amber,
                        shape: BoxShape.circle,
                      ),
                    ),
                    const SizedBox(width: 5),
                    Expanded(
                      child: Text(
                        controller.backendAvailable
                            ? 'Local assistant ready'
                            : 'Local backend',
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: TextStyle(
                          color: Colors.white.withValues(alpha: 0.65),
                          fontSize: 10,
                        ),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
          if (availableModels.isNotEmpty)
            SizedBox(
              width: 96,
              child: DropdownButtonHideUnderline(
                child: DropdownButton<String>(
                  value: availableModels.any(
                    (model) => model.key == controller.selectedModel,
                  )
                      ? controller.selectedModel
                      : availableModels.first.key,
                  isExpanded: true,
                  dropdownColor: AppColors.deepOcean,
                  iconEnabledColor: Colors.white70,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 11,
                    fontWeight: FontWeight.w700,
                  ),
                  items: availableModels
                      .map(
                        (model) => DropdownMenuItem(
                          value: model.key,
                          child: Text(
                            _shortModelName(model.key),
                            overflow: TextOverflow.ellipsis,
                          ),
                        ),
                      )
                      .toList(),
                  onChanged: controller.isStreaming
                      ? null
                      : (value) {
                          if (value != null) {
                            unawaited(controller.setModel(value));
                          }
                        },
                ),
              ),
            ),
          IconButton(
            tooltip: 'Clear conversation',
            onPressed: controller.isStreaming ? null : onClear,
            color: Colors.white70,
            iconSize: 20,
            visualDensity: VisualDensity.compact,
            icon: const Icon(Icons.delete_sweep_outlined),
          ),
          IconButton(
            tooltip: 'Collapse assistant',
            onPressed: controller.close,
            color: Colors.white,
            iconSize: 20,
            visualDensity: VisualDensity.compact,
            icon: const Icon(Icons.close_rounded),
          ),
        ],
      ),
    );
  }

  String _shortModelName(String model) => switch (model) {
        'llama' => 'Llama 3.1',
        'mistral' => 'Mistral',
        'qwen' => 'Qwen3',
        _ => model,
      };
}

class _PredictionContextBanner extends StatelessWidget {
  const _PredictionContextBanner({required this.controller});

  final AssistantController controller;

  @override
  Widget build(BuildContext context) {
    final result = controller.currentPrediction!;
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 9),
      color: AppColors.aqua.withValues(alpha: context.isDarkMode ? 0.12 : 0.08),
      child: Row(
        children: [
          const Icon(Icons.image_search_rounded,
              size: 17, color: AppColors.aqua),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              'Using scan: ${result.disease.name} • '
              '${(result.confidence * 100).toStringAsFixed(1)}%',
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(
                color: context.textPrimary,
                fontSize: 11,
                fontWeight: FontWeight.w700,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _WelcomeView extends StatelessWidget {
  const _WelcomeView({required this.onSuggestion});

  final ValueChanged<String> onSuggestion;

  static const suggestions = [
    'Explain my latest fish disease prediction.',
    'What signs indicate poor water quality?',
    'How can I improve dissolved oxygen?',
  ];

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(22, 28, 22, 16),
      child: Column(
        children: [
          Container(
            width: 62,
            height: 62,
            decoration: BoxDecoration(
              gradient: const LinearGradient(
                colors: [AppColors.seaBlue, AppColors.teal],
              ),
              borderRadius: BorderRadius.circular(20),
            ),
            child: const Icon(
              Icons.auto_awesome_rounded,
              color: Colors.white,
              size: 30,
            ),
          ),
          const SizedBox(height: 16),
          Text(
            'Ask about fish health',
            style: TextStyle(
              color: context.textPrimary,
              fontSize: 19,
              fontWeight: FontWeight.w800,
            ),
          ),
          const SizedBox(height: 7),
          Text(
            'I use your latest scan and the local aquaculture knowledge base. '
            'No question is sent to an external AI API.',
            textAlign: TextAlign.center,
            style: TextStyle(
              color: context.textSecondary,
              fontSize: 12,
              height: 1.45,
            ),
          ),
          const SizedBox(height: 22),
          for (final suggestion in suggestions)
            Padding(
              padding: const EdgeInsets.only(bottom: 9),
              child: InkWell(
                onTap: () => onSuggestion(suggestion),
                borderRadius: BorderRadius.circular(14),
                child: Ink(
                  width: double.infinity,
                  padding:
                      const EdgeInsets.symmetric(horizontal: 13, vertical: 11),
                  decoration: BoxDecoration(
                    color: context.tintedSurface,
                    borderRadius: BorderRadius.circular(14),
                    border: Border.all(color: context.subtleBorder),
                  ),
                  child: Row(
                    children: [
                      const Icon(
                        Icons.arrow_forward_rounded,
                        size: 15,
                        color: AppColors.aqua,
                      ),
                      const SizedBox(width: 9),
                      Expanded(
                        child: Text(
                          suggestion,
                          style: TextStyle(
                            color: context.textPrimary,
                            fontSize: 11,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }
}

class _Composer extends StatelessWidget {
  const _Composer({
    required this.input,
    required this.focusNode,
    required this.isStreaming,
    required this.onSend,
  });

  final TextEditingController input;
  final FocusNode focusNode;
  final bool isStreaming;
  final VoidCallback onSend;

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      top: false,
      child: Padding(
        padding: const EdgeInsets.fromLTRB(10, 9, 10, 10),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            Expanded(
              child: TextField(
                controller: input,
                focusNode: focusNode,
                enabled: !isStreaming,
                minLines: 1,
                maxLines: 4,
                textInputAction: TextInputAction.newline,
                decoration: InputDecoration(
                  hintText: isStreaming
                      ? 'Generating locally...'
                      : 'Ask about this scan or fish health',
                  isDense: true,
                  filled: true,
                  fillColor: context.tintedSurface,
                  contentPadding: const EdgeInsets.symmetric(
                    horizontal: 13,
                    vertical: 11,
                  ),
                ),
              ),
            ),
            const SizedBox(width: 8),
            IconButton.filled(
              tooltip: 'Send message',
              onPressed: isStreaming ? null : onSend,
              icon: isStreaming
                  ? const SizedBox(
                      width: 17,
                      height: 17,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        color: Colors.white,
                      ),
                    )
                  : const Icon(Icons.arrow_upward_rounded),
            ),
          ],
        ),
      ),
    );
  }
}
