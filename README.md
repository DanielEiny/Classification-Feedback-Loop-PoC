# Classification-Feedback-Loop-PoC
### Looking to re-train your classification model in an efficient way? start here.




<p float="center">
 <img src="Resources/EverCompliantLOGO.png" width="300" hspace="50"/> 
 <img src="Resources/BarIlanLOGO.jpg" width="240" hspace="50"/> 
</p>
   
This repo summerize the work we've done in collaboration with EverCompliant company, as a final projet of the course 'Applied Data Science with Python' (BIU, summer 2019 by dr. Omri Allouche). 

[EverCompliant](https://evercompliant.com) is a leading provider of cyber risk intelligence and transaction laundering detection and prevention. One of the algorithms the company uses is a classification model, that categorize web pages according to their textual content. From time to time, this model requires a re-train, in order to improve performance or to be updated with the changing environment. For this training, we need labeled data. But labeling is expensive!
The solution sounds simple: focus on most relevant and important samples, and label only them. But how can one know a priori how important a sample is? The relatively-new field that try to deal with this problem is called 'Active Learn', and in this work we've tried to find the best approaches and methods for EverCompliant scenario. In this repo you will find a summary of this research and its results, as well as the Python code written for this aim.


To get an overview of what we've done, read our [project paper](ProjectPaper.ipynb)!

For a more detailed description of each step, see [Notebooks](Notebooks/README.rm).

And finally, feel free to use/contribute the [Code](Code/README.rm).
