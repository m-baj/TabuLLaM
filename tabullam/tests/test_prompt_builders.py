"""
Tests for prompt builders.
"""

import pytest

from tabullam.base.prompt_builder import TaskMetadata, BasePromptBuilder
from tabullam.prompt_builders import (
    StandardPromptBuilder,
    ConfidencePromptBuilder,
    ProbabilitiesPromptBuilder,
    create_prompt_builder,
)


class TestTaskMetadata:
    """Tests for TaskMetadata dataclass."""

    def test_create_task_metadata(self):
        """Test creating TaskMetadata instance."""
        metadata = TaskMetadata(
            feature_names=['age', 'income'],
            class_labels=['low', 'high'],
            target_name='category',
            task_description='Classify customers'
        )
        assert metadata.feature_names == ['age', 'income']
        assert metadata.class_labels == ['low', 'high']
        assert metadata.target_name == 'category'
        assert metadata.task_description == 'Classify customers'

    def test_create_task_metadata_without_description(self):
        """Test creating TaskMetadata without optional description."""
        metadata = TaskMetadata(
            feature_names=['a', 'b'],
            class_labels=['x', 'y'],
            target_name='target'
        )
        assert metadata.task_description is None


class TestStandardPromptBuilder:
    """Tests for StandardPromptBuilder."""

    @pytest.fixture
    def builder(self, task_metadata):
        """Create a StandardPromptBuilder instance."""
        return StandardPromptBuilder(task_metadata)

    def test_build_zero_shot_prompt(self, builder, sample_features):
        """Test building zero-shot prompt."""
        prompt = builder.build(sample_features)

        # Check structure
        assert '## Instruction' in prompt
        assert '## Task Information' in prompt
        assert '## Query' in prompt
        assert '## Output' in prompt

        # Check features are included
        assert 'age: 35' in prompt
        assert 'income: 50000' in prompt

    def test_build_with_examples(self, builder, sample_features, few_shot_examples):
        """Test building prompt with few-shot examples."""
        prompt = builder.build(sample_features, few_shot_examples)

        assert '## Examples' in prompt
        # Check examples are included
        assert 'age: 25' in prompt  # First example
        assert 'low' in prompt  # First example's label

    def test_prompt_contains_class_labels(self, builder, sample_features):
        """Test that prompt contains class labels."""
        prompt = builder.build(sample_features)
        assert 'low' in prompt
        assert 'medium' in prompt
        assert 'high' in prompt

    def test_prompt_contains_task_description(self, builder, sample_features):
        """Test that prompt contains task description."""
        prompt = builder.build(sample_features)
        assert 'Predict customer spending category' in prompt

    def test_default_instruction(self, builder):
        """Test that default instruction is set."""
        assert 'classification assistant' in builder.instruction.lower()

    def test_custom_instruction(self, task_metadata):
        """Test using custom instruction."""
        custom = "You are a custom classifier"
        builder = StandardPromptBuilder(task_metadata, instruction=custom)
        assert builder.instruction == custom


class TestConfidencePromptBuilder:
    """Tests for ConfidencePromptBuilder."""

    @pytest.fixture
    def builder(self, task_metadata):
        """Create a ConfidencePromptBuilder instance."""
        return ConfidencePromptBuilder(task_metadata)

    def test_build_prompt(self, builder, sample_features):
        """Test building confidence prompt."""
        prompt = builder.build(sample_features)

        # Check JSON format instructions
        assert 'JSON' in prompt
        assert 'prediction' in prompt
        assert 'confidence_score' in prompt

    def test_output_format_section(self, builder, sample_features):
        """Test output format section contains expected elements."""
        prompt = builder.build(sample_features)

        assert '0 to 10' in prompt or '0-10' in prompt
        assert 'JSON' in prompt

    def test_prompt_with_examples(self, builder, sample_features, few_shot_examples):
        """Test building prompt with examples."""
        prompt = builder.build(sample_features, few_shot_examples)
        assert '## Examples' in prompt


class TestProbabilitiesPromptBuilder:
    """Tests for ProbabilitiesPromptBuilder."""

    @pytest.fixture
    def builder(self, task_metadata):
        """Create a ProbabilitiesPromptBuilder instance."""
        return ProbabilitiesPromptBuilder(task_metadata)

    def test_build_prompt(self, builder, sample_features):
        """Test building probabilities prompt."""
        prompt = builder.build(sample_features)

        # Check JSON format instructions
        assert 'JSON' in prompt
        assert 'score' in prompt.lower()

    def test_output_mentions_all_classes(self, builder, sample_features):
        """Test that output section mentions all classes."""
        prompt = builder.build(sample_features)

        # Class names should be in the output format section
        assert '"low"' in prompt or "'low'" in prompt
        assert '"medium"' in prompt or "'medium'" in prompt
        assert '"high"' in prompt or "'high'" in prompt

    def test_prompt_with_examples(self, builder, sample_features, few_shot_examples):
        """Test building prompt with examples."""
        prompt = builder.build(sample_features, few_shot_examples)
        assert '## Examples' in prompt


class TestCreatePromptBuilder:
    """Tests for create_prompt_builder factory function."""

    def test_create_standard_builder(self, task_metadata):
        """Test creating StandardPromptBuilder."""
        builder = create_prompt_builder('standard', task_metadata)
        assert isinstance(builder, StandardPromptBuilder)

    def test_create_confidence_builder(self, task_metadata):
        """Test creating ConfidencePromptBuilder."""
        builder = create_prompt_builder('confidence', task_metadata)
        assert isinstance(builder, ConfidencePromptBuilder)

    def test_create_probabilities_builder(self, task_metadata):
        """Test creating ProbabilitiesPromptBuilder."""
        builder = create_prompt_builder('probabilities', task_metadata)
        assert isinstance(builder, ProbabilitiesPromptBuilder)

    def test_create_with_custom_instruction(self, task_metadata):
        """Test creating builder with custom instruction."""
        custom = "Custom instruction"
        builder = create_prompt_builder('standard', task_metadata, custom)
        assert builder.instruction == custom

    def test_create_invalid_type_raises(self, task_metadata):
        """Test that invalid prompt type raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            create_prompt_builder('invalid_type', task_metadata)
        assert 'Unknown prompt_type' in str(exc_info.value)


class TestPromptStructure:
    """Tests for overall prompt structure and formatting."""

    def test_prompt_sections_are_separated(self, task_metadata, sample_features):
        """Test that prompt sections are properly separated."""
        builder = StandardPromptBuilder(task_metadata)
        prompt = builder.build(sample_features)

        # Sections should be separated by double newlines
        sections = prompt.split('\n\n')
        assert len(sections) >= 4  # At least: Instruction, Task Info, Query, Output

    def test_examples_are_numbered(self, task_metadata, sample_features, few_shot_examples):
        """Test that examples are numbered."""
        builder = StandardPromptBuilder(task_metadata)
        prompt = builder.build(sample_features, few_shot_examples)

        assert '1.' in prompt
        assert '2.' in prompt
        assert '3.' in prompt

    def test_feature_names_in_task_info(self, task_metadata, sample_features):
        """Test that feature names are listed in task info."""
        builder = StandardPromptBuilder(task_metadata)
        prompt = builder.build(sample_features)

        assert 'age' in prompt
        assert 'income' in prompt
        assert 'education' in prompt
