"""Auditable missingness, parser coverage, and training-support diagnostics."""
from __future__ import annotations

from .data_preprocessing import FEATURE_COLUMNS
from .feature_engineering import MEASURE_FIELDS, TOKEN_FIELDS, is_missing, measurement_summary, tokens


def quality_report(frame, training_options, vocabulary):
    """Inspect one explicitly named partition; rates use all rows as denominator."""
    fields = {}
    rows = len(frame)
    measurements = {column: units for column, units in MEASURE_FIELDS.values()}
    token_fields = {column: prefix for prefix, column in TOKEN_FIELDS.items()}
    for column in FEATURE_COLUMNS:
        values = frame[column]
        known = set(training_options[column])
        missing = values.map(is_missing)
        unseen = ~missing & ~values.isin(known)
        entry = {'rows': rows, 'missing_count': int(missing.sum()),
                 'missing_rate': float(missing.mean()) if rows else None,
                 'unknown_category_count': int(unseen.sum()),
                 'unknown_category_rate': float(unseen.mean()) if rows else None}
        if column in measurements:
            parsed = [measurement_summary(v, measurements[column]) for v in values]
            accepted = [sum(p[f'{u}_count'] for u in measurements[column]) for p in parsed]
            entry['measurement'] = {
                'rows_with_parsed_values': sum(n > 0 for n in accepted),
                'rows_with_unparsed_values': sum(p['unparsed_count'] > 0 for p in parsed),
                'rows_with_unspecified_units': sum(p['unitless_count'] > 0 for p in parsed),
                'rows_fully_parsed': sum(n > 0 and not p['partial'] for n, p in zip(accepted, parsed)),
                'parsed_row_rate': sum(n > 0 for n in accepted) / rows if rows else None,
                'unparsed_row_rate': sum(p['unparsed_count'] > 0 for p in parsed) / rows if rows else None,
            }
        if column in token_fields:
            vocab = set(vocabulary[token_fields[column]])
            outside = [len(set(tokens(v)[0]) - vocab) for v in values]
            entry['tokens'] = {'rows_outside_bounded_vocabulary': sum(n > 0 for n in outside),
                               'outside_vocabulary_row_rate': sum(n > 0 for n in outside) / rows if rows else None}
        fields[column] = entry
    return {'rows': rows, 'rate_denominator': 'All rows in this partition; missing values are excluded from unknown-category counts.',
            'vocabulary_notice': 'Outside the bounded token vocabulary does not necessarily mean unseen in training.',
            'fields': fields}


def availability_snapshot(audit):
    """Keep existing measured null counts useful when the source is unavailable."""
    total = audit['cleaned_rows']
    return {
        'status': 'source_data_required',
        'scope': 'All cleaned labeled records, across train/validation/test; literal nulls only.',
        'rows': total,
        'notice': 'Counts below come from the saved Phase 1 audit. Semantic Unknown labels, parser coverage, and partition-specific support rates require the original source CSV.',
        'fields': {c: {'null_count': int(audit['missing_values_before_normalization'][c]),
                       'null_rate': audit['missing_values_before_normalization'][c] / total,
                       'semantic_missing_rate': None, 'parsing_success_rate': None,
                       'unknown_category_rate': None} for c in FEATURE_COLUMNS},
    }
