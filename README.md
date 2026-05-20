# ferrer_ip_project-on-homogenity
stores code results, data, and ipynb for easy reproduction

# how to run
Download ```ferrer_analysis.ipynb``` and run each cell from start to finish, use common sense in adding your own api keys and uncommenting code files that may be commented out during my own running.

also feel free to check out my [collab](https://colab.research.google.com/drive/1mVKVzxDcrmTbAD2b_cQFW2mexg_qo0yB?usp=sharing) notebook for this project.

------------------------------------------------------------------
Through analysis of 900 generated ideas from 10 total prompts and 6 models (5 ideas per prompt), we answer several questions.

Do same models repeat similar ideas across runs, do different models converge on similar ideas, and do some prompts have narrow idea pools? This is similar to Jiang’s study applied to high school ap lang prompts. In a full study, English teachers or human annotators would label each thesis angles and group them. 

Intra-model repetititon was clear. For instance, claude-sonnet-4.6 produced 15 ideas for the “collegeboard5” prompt (the extent to which Baca’s claim about the value of possessions is valid), but they only collpased into 2 distinct clusters of different ideas. Its intra-model reptition rate was 0.867, and the overall rate was 0.706, meaning a large portion of separate outputs by the same model repeated similar, narrow claims instead of generating diverse ideas.

Inter-model homogenity was also clear. For example, for claude-sonnet-4.6 and gemini-2.5-flash for the collegeboard2 prompt, both models shared 5 clsuters out of 5 total clusters, meaning they generated exactly the same ideas. Overall, the inter-model homogenity rate was  0.793, which shows a large portion of cross-model families often used by students online generate similar ideas to high school AP lang prompts.

Lastly, the “ferrer3” prompt on barbara jordan and private wants produced only 3 broad clusters of ideas out of 90 generated ideas. The most common cluster was “threat conditioned by institutions” for 35/90 prompts. This showcases that LLMs for some prompts may severely limit that amount of angles shown to students, in turn possibly impacting their own creativity.

This shows that the inputs high school students draw upon for inspiration, writing, or refinement are often homogenous as well in an AP Lang setting, cluing in our future counterbalanced within-subjects design study with real participants.


