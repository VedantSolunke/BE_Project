# Ablation Study Results
Date: 2025-04-15

## Overview

This report presents the results of an ablation study performed on the Legal Assistance Platform.
The study tested different system configurations to measure the impact of individual components on overall performance.

## Configurations Tested

- **Baseline**
- **Without Llm**
- **Smaller Set**

## Results Summary

| Configuration | Precision | Retrieval Time (s) | Analysis Time (s) | Total Time (s) |
|--------------|-----------|-------------------|------------------|---------------|
| baseline | 0.0000 | 0.0004 | 6.9911 | 6.9916 |
| without_llm | 0.0000 | 0.0002 | 0.0000 | 0.0002 |
| smaller_set | 0.0000 | 0.0005 | 6.7288 | 6.7292 |

## Key Findings

- The configuration with the highest precision was **baseline** (0.0000).
- The fastest configuration was **without_llm** (0.0002 seconds).

### Impact of LLM Analysis

- Removing LLM analysis changed precision by 0.0000 (decreased).
- Removing LLM analysis saved 6.9914 seconds per query.

### Impact of Result Set Size

- Using a smaller result set changed precision by 0.0000 (decreased).
- Using a smaller result set saved 0.2623 seconds per query.

## Conclusion

This ablation study provides insights into the contribution of each system component to overall performance.
These findings can guide future optimization efforts and help balance precision and speed requirements.
