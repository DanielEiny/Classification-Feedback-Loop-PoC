# Classification-Feedback-Loop-PoC
Looking to re-train your classification model in an efficient way? you are in the right place.




<p float="center">
 <img src="Resources/EverCompliantLOGO.png" width="300" hspace="50"/> 
 <img src="Resources/BarIlanLOGO.jpg" width="240" hspace="50"/> 
</p>
   
This repo summerize the work we've done in collaboration with EverCompliant company, as a final projet of the course 'Applied Data Science with Python' (BIU, summer 2019 by dr. Omri Allouche). 

[EverCompliant](https://evercompliant.com) is a leading provider of cyber risk intelligence and transaction laundering ldtection and prevention. One of the major tools that the company uses is a classification model, that categorize web pages according to their textual content. For time to time, this model requairs are-train, in order to improve preformance or to be update with the changing environment. For this training, we need labaled data. But labeling is expensive!
The solution sound simple - focus on most relevant and important samples, and label only them. But how can we know a priori how important is a sample? The relatively-new field that try to deal with this problam is called 'Active Learn', and in this work we've tried to find the best approaches and methods for EverCompliant scenario. In this repo you will find a summery of this research and its results, as well as the Python code written for this aim.

In this work, we made a small research of methods to make the 

To get an overview of what we've done, read our [project paper](ProjectPaper.ipynb)!

For a more detailed description of each step, see [Notebooks](Notebooks/README.rm).

And finally, feel free to use/contribute the [Code](Code/README.rm).
