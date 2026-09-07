# System Architecture

## High-Level Architecture

The following diagram represents the proposed architecture of the
Autonomous Data Analysis Agent.


---

## 1. User

The system accepts structured data from users through multiple sources:

- CSV
- Excel
- Database

The user provides the dataset that needs to be analyzed.

---

## 2. Data Ingestion Layer

The Data Ingestion Layer is responsible for loading data from
different sources into the analysis pipeline.

### Responsibilities

- CSV ingestion
- Excel ingestion
- Database connectivity
- Data normalization
- Input validation
- Dataset registration

---

## 3. Data Profiling

The Data Profiling component analyzes the basic characteristics of
the dataset.

### Responsibilities

- Schema detection
- Data type identification
- Missing value analysis
- Duplicate detection
- Basic statistics
- Cardinality analysis
- Data quality checks

### Output

A structured data profile containing information about the dataset,
columns, data types, statistics, and quality issues.

---

## 4. Data Understanding Agent

The Data Understanding Agent determines the semantic meaning of
the dataset.

### Responsibilities

- Understand column meanings
- Identify potential business entities
- Identify candidate keys
- Understand categorical and numerical fields
- Identify possible relationships

---

## 5. Relationship Discovery Agent

This agent discovers relationships between datasets.

### Relationship Types

- Table ↔ Table
- Dataset ↔ Dataset
- Column ↔ Column

### Responsibilities

- Identify potential foreign keys
- Detect matching columns
- Compare schemas
- Identify join possibilities
- Determine relationship confidence

---

## 6. Analysis Planner Agent

The Analysis Planner determines what analysis should be performed.

It answers:

> "What should I investigate in this data?"

The planner generates an analysis plan based on:

- Dataset characteristics
- Business context
- Relationships
- Data quality
- Potential analytical opportunities

---

## 7. Specialized Analysis Agents

The analysis plan is distributed to specialized agents.

### Trend Agent

Identifies:

- Time-based trends
- Growth/decline
- Seasonality
- Changes over time

### Correlation Agent

Identifies:

- Relationships between variables
- Correlation patterns
- Strong and weak associations

### Anomaly Agent

Identifies:

- Outliers
- Unexpected behavior
- Abnormal patterns
- Potential data quality issues

---

## 8. Hypothesis Agent

The Hypothesis Agent generates and tests potential explanations
for observed patterns.

### Responsibilities

- Generate hypotheses
- Define supporting evidence
- Test hypotheses against available data
- Reject unsupported hypotheses

---

## 9. Validation Agent

The Validation Agent verifies whether generated insights are reliable.

### Statistical Validation

- Statistical significance
- Confidence
- Sample size
- Appropriate statistical tests

### Data Quality Validation

- Missing data impact
- Duplicate impact
- Data consistency
- Reliability of source data

---

## 10. Insight Ranking Agent

Not every discovered insight is equally valuable.

The Insight Ranking Agent ranks insights based on:

- Confidence
- Business impact
- Novelty
- Evidence strength
- Relevance

---

## 11. Report Generation Agent

The Report Generation Agent converts validated insights into
a human-readable report.

### Output

- Key findings
- Explanation
- Charts
- Supporting evidence
- Statistical results
- Confidence information

---

## 12. Final Output

The final output contains:

- Insights
- Charts
- Evidence
- Validated findings
- Analytical summary
