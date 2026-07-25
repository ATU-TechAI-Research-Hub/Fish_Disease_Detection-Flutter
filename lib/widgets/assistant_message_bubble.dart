import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_markdown_plus/flutter_markdown_plus.dart';

import '../models/assistant_message.dart';
import '../theme/app_theme.dart';

class AssistantMessageBubble extends StatelessWidget {
  const AssistantMessageBubble({
    required this.message,
    this.onRegenerate,
    super.key,
  });

  final AssistantMessage message;
  final VoidCallback? onRegenerate;

  @override
  Widget build(BuildContext context) {
    final isUser = message.role == AssistantRole.user;
    final color = message.isError
        ? AppColors.coral.withValues(alpha: 0.09)
        : isUser
            ? AppColors.seaBlue
            : context.surfaceCard;
    final foreground = isUser ? Colors.white : context.textPrimary;

    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        constraints: const BoxConstraints(maxWidth: 560),
        margin: EdgeInsets.only(
          left: isUser ? 42 : 0,
          right: isUser ? 0 : 18,
          bottom: 14,
        ),
        padding: const EdgeInsets.fromLTRB(14, 12, 14, 9),
        decoration: BoxDecoration(
          color: color,
          borderRadius: BorderRadius.only(
            topLeft: const Radius.circular(18),
            topRight: const Radius.circular(18),
            bottomLeft: Radius.circular(isUser ? 18 : 5),
            bottomRight: Radius.circular(isUser ? 5 : 18),
          ),
          border: isUser
              ? null
              : Border.all(
                  color: message.isError
                      ? AppColors.coral.withValues(alpha: 0.3)
                      : context.subtleBorder,
                ),
          boxShadow: isUser
              ? null
              : [
                  BoxShadow(
                    color: context.cardShadow,
                    blurRadius: 8,
                    offset: const Offset(0, 3),
                  ),
                ],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (!isUser) ...[
              Row(
                children: [
                  Container(
                    width: 24,
                    height: 24,
                    decoration: BoxDecoration(
                      gradient: const LinearGradient(
                        colors: [AppColors.seaBlue, AppColors.teal],
                      ),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: const Icon(
                      Icons.auto_awesome_rounded,
                      color: Colors.white,
                      size: 14,
                    ),
                  ),
                  const SizedBox(width: 8),
                  Text(
                    'AquaScan AI',
                    style: TextStyle(
                      color: foreground,
                      fontSize: 12,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                  if (message.model != null) ...[
                    const Spacer(),
                    Text(
                      message.model!.toUpperCase(),
                      style: TextStyle(
                        color: context.textSecondary,
                        fontSize: 9,
                        fontWeight: FontWeight.w700,
                        letterSpacing: 0.6,
                      ),
                    ),
                  ],
                ],
              ),
              const SizedBox(height: 9),
            ],
            if (message.isStreaming && message.content.isEmpty)
              const _TypingIndicator()
            else if (isUser)
              SelectableText(
                message.content,
                style: TextStyle(
                  color: foreground,
                  fontSize: 14,
                  height: 1.45,
                ),
              )
            else
              MarkdownBody(
                data: message.content,
                selectable: true,
                styleSheet: MarkdownStyleSheet.fromTheme(
                  Theme.of(context),
                ).copyWith(
                  p: TextStyle(color: foreground, fontSize: 14, height: 1.5),
                  h1: TextStyle(
                    color: foreground,
                    fontSize: 20,
                    fontWeight: FontWeight.w800,
                  ),
                  h2: TextStyle(
                    color: foreground,
                    fontSize: 17,
                    fontWeight: FontWeight.w800,
                  ),
                  h3: TextStyle(
                    color: foreground,
                    fontSize: 15,
                    fontWeight: FontWeight.w800,
                  ),
                  listBullet:
                      TextStyle(color: foreground, fontSize: 14, height: 1.5),
                  code: TextStyle(
                    color: context.isDarkMode
                        ? AppColors.seafoam
                        : AppColors.ocean,
                    backgroundColor: context.tintedSurface,
                    fontSize: 12,
                  ),
                ),
              ),
            if (!isUser && message.sources.isNotEmpty) ...[
              const SizedBox(height: 10),
              Wrap(
                spacing: 6,
                runSpacing: 6,
                children: [
                  for (int i = 0; i < message.sources.length; i++)
                    Tooltip(
                      message: message.sources[i].source,
                      child: Container(
                        constraints: const BoxConstraints(maxWidth: 240),
                        padding: const EdgeInsets.symmetric(
                          horizontal: 8,
                          vertical: 4,
                        ),
                        decoration: BoxDecoration(
                          color: context.tintedSurface,
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Text(
                          '[${i + 1}] ${message.sources[i].title}',
                          overflow: TextOverflow.ellipsis,
                          style: TextStyle(
                            color: context.textSecondary,
                            fontSize: 10,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ),
                    ),
                ],
              ),
            ],
            if (!isUser && !message.isStreaming && message.content.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(top: 5),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.end,
                  children: [
                    _ActionIcon(
                      tooltip: 'Copy response',
                      icon: Icons.copy_rounded,
                      onPressed: () async {
                        await Clipboard.setData(
                          ClipboardData(text: message.content),
                        );
                        if (context.mounted) {
                          ScaffoldMessenger.maybeOf(context)?.showSnackBar(
                            const SnackBar(
                              content: Text('Response copied'),
                              duration: Duration(seconds: 1),
                            ),
                          );
                        }
                      },
                    ),
                    if (onRegenerate != null)
                      _ActionIcon(
                        tooltip: 'Regenerate response',
                        icon: Icons.refresh_rounded,
                        onPressed: onRegenerate!,
                      ),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class _ActionIcon extends StatelessWidget {
  const _ActionIcon({
    required this.tooltip,
    required this.icon,
    required this.onPressed,
  });

  final String tooltip;
  final IconData icon;
  final VoidCallback onPressed;

  @override
  Widget build(BuildContext context) {
    return IconButton(
      tooltip: tooltip,
      onPressed: onPressed,
      visualDensity: VisualDensity.compact,
      iconSize: 16,
      color: context.textSecondary,
      icon: Icon(icon),
    );
  }
}

class _TypingIndicator extends StatefulWidget {
  const _TypingIndicator();

  @override
  State<_TypingIndicator> createState() => _TypingIndicatorState();
}

class _TypingIndicatorState extends State<_TypingIndicator>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 900),
  )..repeat();

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, _) {
        return Row(
          mainAxisSize: MainAxisSize.min,
          children: List.generate(3, (index) {
            final phase = (_controller.value - index * 0.18) % 1.0;
            final opacity = 0.35 + (1 - (phase - 0.5).abs() * 2) * 0.65;
            return Container(
              width: 7,
              height: 7,
              margin: const EdgeInsets.only(right: 5),
              decoration: BoxDecoration(
                color: AppColors.aqua.withValues(alpha: opacity),
                shape: BoxShape.circle,
              ),
            );
          }),
        );
      },
    );
  }
}
