import typer
import pandas as pd
from message.data import transform_features_py  # noqa
from message.data import get_features
from message.model import ChatModel, OpenAIKeys

print(OpenAIKeys)

app = typer.Typer()


@app.command()
def transform() -> None:
    """Deploy transformation into exercise results."""

    transform_features_py()

    return


@app.command()
def get_message(session_group: str) -> None:
    """Generate an AI message based on a specific session's
     data and asks PT for a decision upon that.
     PT can accept, edit or reject the message.

    Parameters
    ----------
    session_group : str
        Specific session group identifier string.
    """

    chat_model = ChatModel()
    features = get_features(session_group=session_group)[0]

    system_message = (
        f"""You are a physical therapist assistant that crafts highly personalized messages related to 
        patient therapy session results. Session results are a collection of datapoints that you will be given access to.

        The objective of the message is to acknowledge completion of the session, reinforce communication and consistency
        and keep the patient engaged in the process.

        Each session is generally classified as OK or Not OK. When the session is Not Okay, you must downplay any technical 
        or clinical issues and generate conversation to problem solve, taking into account negative datapoints of the session.

        Here is an example that fits our message standards for an Ok session:
           Fantastic job completing your session Matt!
           I’m curious, how do you feel your first session went?
           To kick-start your progress, I suggest a session every other day for the next two weeks.
           How does this plan sound? Will that work for you?

        And this is an example that fits our message standards for a Not Okay session:"
           Fantastic job completing your session, Matt!
           👏 That is a huge win, so give yourself a pat on the back!
           I reviewed your results, and it looks like you may have had a bit of trouble with the hip raise exercise.
           This can happen in the first session or two as the system gets used to the way you move.
           Can you tell me a little bit about what happened here? Was tech the issue on that one?
        
        Finally, you do not use numbers, you express them as quantities.
        """
    )

    # Useful session information and context for user message - column choice variables.

    # General information
    therapy_name = features['therapy_name']
    patient_name = features['patient_name']
    session_is_nok = features['session_is_nok']
    session_number = features['session_number']
    pain = features['pain']
    fatigue = features['fatigue']
    quality = features['quality']

    # Technical agglomeration
    movement_detection = features['quality_reason_movement_detection']
    tablet = features['quality_reason_tablet']
    trackers = features['quality_reason_tablet_and_or_motion_trackers']
    system_problems = features['leave_exercise_system_problem']

    # Specific information
    perc_correct_repeats = features['perc_correct_repeats']
    leave_session = features['leave_session']
    exercise_with_most_incorrect = features['exercise_with_most_incorrect']

    user_message = (
        f"""Please generate a message for the following session results.

        Guidelines to formatting:
        - Length: Max of 4 sentences. Each paragraph seperated.
        - Tone: Laid-back, no clinical language.
        - Patient name awareness: Reference the patients name who participated in the session.
        - Concise, motivational, empathetic. 
        - Max of 1 positive success related emoji.
        - Do not ask questions in the middle of the message. Conclude with a single open-ended question.
        - No formal goodbyes.
        - Break down message into manageable pieces.

         {therapy_name} therapy session results of {patient_name}:
         The session was {'Not OK' if session_is_nok else 'OK'}.
         It was session number {session_number} of this patient.
         The patient reported a pain level of {pain}, fatigue level of {fatigue}.
         The feedback on the session general quality (from 1 to 5) was {quality}, with {'no' if (pd.notna(quality) & quality == 5) else ''}
         specific technical hiccups {'due to movement_detection,' if movement_detection else ''}
          {'due to tablet issues,' if tablet else ''} {'due to tablet and/or motion tracking' if trackers else ''}.
          The patient had {system_problems} technical issues in specific exercises.
         {perc_correct_repeats*100}% of exercise repetitions were flawless. 
         The patient {'did not' if leave_session is None else 'did'} abandon the session.
         For the exercise with most incorrect repetitions, there was {exercise_with_most_incorrect}. """
    )

    # Create the message based on system and user messages.
    message = chat_model.get_completion(
        temperature=0.2,
        model="gpt-3.5-turbo",
        messages=[
            {OpenAIKeys.ROLE: "system", OpenAIKeys.CONTENT: system_message},
            {OpenAIKeys.ROLE: "user", OpenAIKeys.CONTENT: user_message}
        ],
        max_tokens=100,
        frequency_penalty=1,
    )

    # Present the AI-generated message to the PT.
    typer.echo(f"Suggestion:\n{message}")

    # Present possible decision options to the PT.
    options = {'Accept': accept, 'Edit': edit, 'Reject': reject}
    selected_option = typer.prompt(text=f"Please choose one of the following options:\n {', '.join(options.keys())}")

    # Select and apply outcome functions. Prepared to handle different case-sensitive approaches.
    if selected_option.lower() in [option.lower() for option in options.keys()]:
        options[selected_option.capitalize()](message)
    else:
        typer.echo(f"Invalid option: {selected_option}")


def accept(message: str) -> None:
    """Accept and send the AI-generated message to the patient.

    Parameters
    ----------
    message : str
        AI-generated message related to the session.
    """

    typer.echo(f"\nMessage sent:\n{message}.")


def edit(message: str) -> None:
    """Edit and send the AI-generated message,
    being able to select a reason and provide extra feedback.

    Parameters
    ----------
    message : str
        AI-generated message related to the session.
    """

    reasons = ["Tone", "Generic", "Engagement", "Factuality", "Other"]
    reason = typer.prompt(text=f"Which of the following editing reasons apply: {reasons}?", default='')
    feedback = typer.prompt(text="Provide additional feedback", default='')
    edited_message = typer.prompt(text=f"\n{message}\nEdit this message")

    typer.echo(f"\nMessage edited. Saved into the system.\nReason: {reason}\nFeedback: {feedback}")

    accept(edited_message)


def reject(message: str) -> None:
    """Rejects the AI-generated message, being able to
     select a reason and provide extra feedback.

    Parameters
    ----------
    message : str
        AI-generated message related to the session.
    """

    reasons = ["Tone", "Generic", "Engagement", "Factuality", "Other"]
    reason = typer.prompt(text=f"Which of the following rejecting reasons apply: {reasons}?", default='')
    feedback = typer.prompt(text="Provide additional feedback", default='')

    typer.echo(f"Message rejected. Saved into the system.\nReason: {reason}\nFeedback: {feedback}")


if __name__ == "__main__":
    app()

