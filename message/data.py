from pathlib import Path

import duckdb
import pandas as pd
from message.config import DATA_DIR, QUERIES_DIR


def open_query(query_filename: Path, **kwargs) -> str:
    """Opens a query file and formats it with the provided kwargs.

    Parameters
    ----------
    query_filename : Path
        Name of the query file to open.

    Returns
    -------
    str
        The query file content formatted with the provided kwargs.
    """
    return open(query_filename, "r").read().format(**kwargs)


def transform_features_py() -> None:
    """Loads exercise results and transforms them
    such that each row is indexed by its session,
    while also creating the necessary new features.

    See report.pdf for more information.
    """

    original_data = pd.read_parquet(
        Path(DATA_DIR, "exercise_results.parquet")
    )

    exercise_independent_ordered_columns = [
        'patient_id', 'patient_name', 'patient_age', 'pain', 'fatigue',
        'therapy_name', 'session_number', 'leave_session', 'quality',
        'quality_reason_movement_detection', 'quality_reason_my_self_personal',
        'quality_reason_other', 'quality_reason_exercises', 'quality_reason_tablet',
        'quality_reason_tablet_and_or_motion_trackers', 'quality_reason_easy_of_use',
        'quality_reason_session_speed', 'session_is_nok'
    ]

    exercise_independent_df = original_data.groupby('session_group').agg({
        **dict.fromkeys(exercise_independent_ordered_columns, 'first')
    })

    exercise_dependent_df = (
        original_data.groupby('session_group', group_keys=True).apply(create_exercise_dependent_columns)
    )
    exercise_dependent_df = exercise_dependent_df.groupby('session_group').first()

    transformed_data = pd.concat([exercise_independent_df, exercise_dependent_df], axis=1).reset_index()
    transformed_data.to_parquet(Path(DATA_DIR, "features_expected.parquet"))


def create_exercise_dependent_columns(group: pd.DataFrame) -> pd.DataFrame:
    """ Auxiliary function for creating exercise dependent columns
    and return their dataframe.

    See report.pdf for more information.

    Parameters
    ----------
    group : pd.DataFrame
        Data containing exercise results grouped by session.

    Returns
    -------
    pd.DataFrame
        Data with exercise-dependent columns transformed and added.
    """

    leave_exercise_system_problem = (group['leave_exercise'] == 'system_problem').sum()
    leave_exercise_other = (group['leave_exercise'] == 'other').sum(),
    leave_exercise_unable_perform = (group['leave_exercise'] == 'unable_perform').sum(),
    leave_exercise_pain = (group['leave_exercise'] == 'pain').sum(),
    leave_exercise_tired = (group['leave_exercise'] == 'tired').sum(),
    leave_exercise_technical_issues = (group['leave_exercise'] == 'technical_issues').sum()
    leave_exercise_difficulty = (group['leave_exercise'] == 'difficulty').sum()

    prescribed_repeats = group['prescribed_repeats'].sum(),
    training_time = group['training_time'].sum(),

    has_correct_reps, has_wrong_reps = group['correct_repeats'].notna().any(), group['wrong_repeats'].notna().any()
    n_correct_repeats = group['correct_repeats'].sum() if has_correct_reps else 0
    n_wrong_repeats = group['wrong_repeats'].sum() if has_wrong_reps else 0
    perc_correct_repeats = n_correct_repeats / (n_correct_repeats + n_wrong_repeats) if has_correct_reps else 0

    number_exercises = (group['training_time'] > 0).sum()
    number_of_distinct_exercises = group[group['training_time'] > 0]['exercise_name'].nunique()

    # Bonus points columns
    exercise_wrong_repeats = group.groupby('exercise_name')['wrong_repeats'].sum()
    exercise_with_most_incorrect = exercise_wrong_repeats.idxmax() if exercise_wrong_repeats.max() > 0 else None

    skipped_exercises = group[group['leave_exercise'].notnull()]
    first_exercise_skipped = skipped_exercises.sort_values('exercise_order')['exercise_name'].iloc[
        0] if not skipped_exercises.empty else None

    return pd.DataFrame({
        'leave_exercise_system_problem': leave_exercise_system_problem,
        'leave_exercise_other': leave_exercise_other,
        'leave_exercise_unable_perform': leave_exercise_unable_perform,
        'leave_exercise_pain': leave_exercise_pain,
        'leave_exercise_tired': leave_exercise_tired,
        'leave_exercise_technical_issues': leave_exercise_technical_issues,
        'leave_exercise_difficulty': leave_exercise_difficulty,
        'prescribed_repeats': prescribed_repeats,
        'training_time': training_time,
        'perc_correct_repeats': perc_correct_repeats,
        'number_exercises': number_exercises,
        'number_of_distinct_exercises': number_of_distinct_exercises,
        'exercise_with_most_incorrect': exercise_with_most_incorrect,
        'first_exercise_skipped': first_exercise_skipped
    }, index=group.index)


def get_features(session_group: str) -> dict:
    """Gets the features for a given session group.

    Parameters
    ----------
    session_group : str
        Session group to filter the features.

    Returns
    -------
    dict
        The features for the given session group in a dict format.
    """
    session = pd.read_parquet(Path(DATA_DIR, "features_expected.parquet"))

    return session[session["session_group"] == session_group].to_dict(
        orient="records"
    )
