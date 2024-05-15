## LLM Workout feedback based on compiled exercise feature engineering. 

Using randomly generated exercise data, I worked on feature engineering, grouping the exercises into workout session rows while also creating additional useful columns that help the model contextualize and provide feedback about overall process.

/message/data - My chosen transformations.

/message/main - A CLI app where I created my system and user prompts based on good practices and guidelines.

Although I used OpenAI models (keys should go on .env), the type of feedback I'm going for is simple enough to use local models, which is something I might develop later on. Finetuning is also possible if I am able to create a lot of good feedback examples, assuming smaller models can also have smaller context windows that don't fit all the exemples I want to use on system.


#### Examples of workout feedback:
![tests](https://github.com/seyeint/Workout_Feedback_DS_LLM/assets/36778187/e4894cf0-152a-4b36-9ed2-e950c0811c60)
