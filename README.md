# DCSC

DCSC is a implemention of paper 'Domain- and category-style clustering for general fake news detection via contrastive learning', and now is under review by Informanction Processing and Management.

##Abstract
Nowadays, online social networks increase information dissemination but also accelerate the spread of fake news. Existing work mainly focus on detecting fake news in a predefined scenario, therefore struggling to handle general tasks, especially the newly emerged events and unseen news domains. Exploration on linguistic styles have shown promising results. However, Most of them require complex preprocessing for capturing styles and ignore the compatibility of certain styles across news domains, and hence are inefficient in real applications. To address these problems, we propose a domain- and category-style clustering framework to learn general style patterns across new domains. Two key modules, content integrity detection (CID) and contrastive style detection (CSD) cooperate to obtain event-independent styles in an adversarial manner, which eliminates the need for data preprocessing. Meanwhile, in the CSD module, a multilevel contrastive loss is developed to perform fine-grained style clustering at both domain and category levels, improving generalization and discrimination of the learned style patterns. Extensive experiments show that our framework improves F1 scores of 2.37\%/2.08\% on the unseen event/news domain, and 0.63\%/1.19\% on the known events/ news domains. Furthermore, the quantitative analysis demonstrates the existence of general style patterns and suggests that real news is more likely to use the hashtag (‘【】’), mention function (@), and numerals, while fake news tends to use ‘!’ and ‘?’. Our code will be available upon acceptance.


##Dataset
The datasets we used in our paper are Pheme and Weibo. We provide the link to the original dataset and data processing code. The Pheme dataset can be download from https://figshare.com/articles/dataset/PHEME_dataset_of_rumours_and_non-rumours/4010619. the weibo dataset is avalable at https://github.com/ICTMCG/Characterizing-Weibo-Multi-Domain-False-News.

 
 The code and details will coming soon. 


 Ackonwledgement

