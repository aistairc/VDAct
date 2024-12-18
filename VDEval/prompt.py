VDEVAL_INSTRUCTION = """
You are given a summary of activities in the video, a question, a set of gold-standard reference answers written by experts, and a candidate answer.
Please rate the accuracy of the candidate answer for the question considering the reference answers, dialogue history, and the summary of activities in the video.
Use a scale of 1-3, with 1 indicating an incorrect or irrelevant answer, 2 indicating an ambiguous or incomplete answer, and 3 indicating a correct answer. Give the rationale before rating. Give rating after 'So rating='.

Summary: The person is in the kitchen, where he interacts with various items. He first walks to and grabs a frying pan from the stove. Then, he goes to the kitchen counter to get a chicken and moves to the sink next to him. Afterward, he grabs a water glass from the kitchen table. The person then goes to the living room, where he interacts with a sofa and a TV before returning to the kitchen. He grabs a salmon from the kitchen counter, soaks it at the sink, and places it on the frying pan on the stove, which he attempts to turn on and off. Finally, he goes back to the kitchen table to grab and place a bottle of wine.

Question: What is the man seeing doing at the very start?
Reference answer: He is seen getting ready to cook a chicken in the kitchen.
Candidate answer: He is in the kitchen and he is grabbing a box of cereal.
Output: The information that he was initially in the kitchen is accurate; however, the detail about him grabbing a box of cereal does not align with the reference and is not included in the summary. Therefore, this answer is only partially correct. So rating=2

Question: How does the chicken turn out?
Reference answer: That is unclear because the man trips and falls.
Candidate answer: He never gets to cook it as he falls flat on his face.
Output: The information \"he falls\" is consistent with the reference, and the statement \"never gets to cook it\" does not contradict the summary. Therefore, this answer is correct. So rating=3
Question: Holding the chicken?
Reference answer: Yes, and a frying pan.
Candidate answer: Yes, he was holding the chicken.
Output: Although the answer does not fully align with the reference, the answer is sufficiently accurate and appropriately addresses the question. So rating=2

Question: Does the man attempt to cook something else after this?
Reference answer: Yes, but not until later on.
Candidate answer: Yes, he does.
Output: The answer correctly responds with 'Yes,' but it omits the detail 'but not until later on,' which is needed to address the 'something else after this' part in reference to the previous dialogue turns. Therefore, the answer is partially correct. So rating=2

Question: What does he do after the fall?
Reference answer: He moves a water glass around on the kitchen table.
Candidate answer: He goes to the living room to relax on the sofa.
Output: The information about relaxing on the sofa in the living room cannot be verified from the summary, the previous dialogue turns, or the reference. Therefore, the answer is incorrect. So rating=1


"""
