enum AssistantRole { user, assistant }

class AssistantSource {
  const AssistantSource({
    required this.title,
    required this.source,
    required this.score,
  });

  final String title;
  final String source;
  final double score;

  factory AssistantSource.fromJson(Map<String, dynamic> json) {
    final rawScore = json['score'];
    return AssistantSource(
      title: json['title']?.toString() ?? 'Aquaculture reference',
      source:
          json['source_name']?.toString() ?? json['source']?.toString() ?? '',
      score: rawScore is num
          ? rawScore.toDouble()
          : double.tryParse(rawScore?.toString() ?? '') ?? 0,
    );
  }

  Map<String, dynamic> toJson() => {
        'title': title,
        'source': source,
        'score': score,
      };
}

class AssistantMessage {
  const AssistantMessage({
    required this.id,
    required this.role,
    required this.content,
    required this.createdAt,
    this.model,
    this.sources = const [],
    this.isStreaming = false,
    this.isError = false,
  });

  final String id;
  final AssistantRole role;
  final String content;
  final DateTime createdAt;
  final String? model;
  final List<AssistantSource> sources;
  final bool isStreaming;
  final bool isError;

  factory AssistantMessage.fromJson(Map<String, dynamic> json) {
    final rawSources = json['sources'] as List<dynamic>? ?? const [];
    return AssistantMessage(
      id: json['id']?.toString() ?? '',
      role: json['role']?.toString() == 'user'
          ? AssistantRole.user
          : AssistantRole.assistant,
      content: json['content']?.toString() ?? '',
      createdAt:
          DateTime.tryParse(json['created_at']?.toString() ?? '')?.toLocal() ??
              DateTime.now(),
      model: json['model']?.toString(),
      sources: rawSources
          .whereType<Map<String, dynamic>>()
          .map(AssistantSource.fromJson)
          .toList(growable: false),
    );
  }

  AssistantMessage copyWith({
    String? id,
    String? content,
    String? model,
    List<AssistantSource>? sources,
    bool? isStreaming,
    bool? isError,
    DateTime? createdAt,
  }) {
    return AssistantMessage(
      id: id ?? this.id,
      role: role,
      content: content ?? this.content,
      createdAt: createdAt ?? this.createdAt,
      model: model ?? this.model,
      sources: sources ?? this.sources,
      isStreaming: isStreaming ?? this.isStreaming,
      isError: isError ?? this.isError,
    );
  }
}

class AssistantModelOption {
  const AssistantModelOption({
    required this.key,
    required this.displayName,
    required this.available,
    required this.active,
  });

  final String key;
  final String displayName;
  final bool available;
  final bool active;

  factory AssistantModelOption.fromJson(Map<String, dynamic> json) {
    return AssistantModelOption(
      key: json['key']?.toString() ?? '',
      displayName: json['display_name']?.toString() ?? '',
      available: json['available'] == true,
      active: json['active'] == true,
    );
  }
}
