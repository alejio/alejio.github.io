---
layout: post
title: Hiring Machine Learning Engineers Part 2 - Job spec and assignment
categories: [Data Science]

---

[Part 1](https://alexiospanos.com/hiring-machine-learning-engineers-part-1/) discussed the emergence of the Machine Learning Engineer role, its distinction from the Data Scientist and the importance of organisations prioritising hiring the former against the latter when data infrastructure maturity is low. In Part 2, we will cover more practical considerations around the hiring process.

## Get the job spec right

If you've spent a bit of time looking at Data Scientist job listings, you'll have certainly encountered specs containing an incoherent mess of random buzzwords:

- "comfortable with Big Data technologies such as Hadoop"
- A bunch of overly prescriptive statistical/Machine Learning techniques (e.g. "k-means clustering")
- A BI/Analytics mention (Tableau usually) 
- A/B testing (always manages to sneak in somehow)
- AI

Of course these job specs are a major turn-off for anyone with experience in the field; a clear tell that the business have no idea why they need such a hire. 

For Machine Learning Engineer specs in the wild, whilst I confess to not having seen as many, I've formed an impression that they are refreshingly less buzzwordy. However, there seems to be a tendency to overstate required skills/aptitudes on the modelling side. 

As we've seen, Machine Learning Engineers operate across the stack and care about all those things that make modelling possible. These include:

- Setting up data pipelines and setting up the storage layers;
- designing APIs;
- orchestrating microservices;
- monitoring performance (operational and predictive);
- managing deployments in the various production environments;
- contributing to different engineering libraries, which can be written in a non "ML-native" programming language;
- finding and fixing production bugs;
- helping Data Scientists write more performant code.

There is so much going on for a Machine Learning Engineer to worry about, so when I see a relevant job spec that mentions Kaggle competitions and implementing arxiv NLP papers, I become a bit sceptical. 

My general opinion for most roles, is that overly specific requirements do not benefit the hiring process.

It is a much better bet to describe in a bit more detail the actual responsibilities of the role rather than bloat the required experience and skills with redundant jargon. In a relevant tangent, I would strongly question the practical value of asking for specific academic qualifications. Does someone actually need a PhD to be a Machine Learning Engineer? In most cases it sounds like more of a cultural (or even vanity) requirement rather than a technical one.

## Getting the assignment right

Take-home assignments, when done right, provide value to both sides of the process regardless of the eventual outcome. 

Nowadays, most companies include this step in the hiring process - and certainly a good candidate solution really boosts the chances of getting an offer! On the other hand, the assignment provides a great signal for the candidate to get a feel for the culture and tech maturity of the business. 

In my opinion the assignment should be designed to give the hiring manager an insight into the candidate's strengths and weaknesses, to further explore in a next stage. Its purpose should not be to extensively probe into all requirements of the role, and should be sufficiently open-ended to allow for originality and creativity to show. Also the trade-offs of making it specific to the business' use case should be considered. Assessors will obviously be much more familiar with the nuances of the business problem, making the "battle" unfair for good candidates who don't happen to have  that specific domain experience. An assignment with a more accessible use case that looks at the core skills required for the role could be more suitable in this sense, and an additional benefit is that it allows the candidate to put it up on their personal git page.

Specifically for Machine Learning Engineers, I believe a very suitable assignment is one that focuses on the end-to-end productionisation of a Machine Learning pipeline.  By stressing in the description that predictive performance is not the main objective of the assignment, it enables the candidate to demonstrate:

- Knowledge in implementing basic Machine Learning pipelines;
- knowledge of software engineering practices such as testing, OOP, containerisation, logging and API/ CLI development.

lA benefit of this type of task is evaluating candidates of different backgrounds; for software engineers moving in to the Machine Learning realm, the assessor can focus a bit more on the Machine Learning potential, whereas for Data Scientists moving into engineering, the assessor focuses on the software engineering potential. In the end, I think that this type of task can be very insightful and educational for more junior candidates, and fairly comfortable for more experienced candidates to show off their knowledge. 

But above all, the assignment should respect the candidate both in terms of their time and their experience. In many cases, candidates will be applying to more than one places and also be in full-time employment. Also, the business must have faith in the hiring process. The best practice is to send out the assignment after recruiters screen the suitability of the candidate - the business must be in a position to trust the recruiters in their decision and not defer basic screening questions to the take-home assignment, as this can come across as very patronising to the candidate.

I've realised that a few assignments sent to me over the years were arguably inadequate. One example that stands out in my memory was for a unicorn scale-up; the recruiter forwarded a case study testing "verbal and numeric reasoning" and containing 5 sections on business understanding, ETL, statistics and coding. Each section had a points system and contained specific questions, which only took one correct answer. Having been working in the industry for a few years already, I thought this kind of test was too much; it was the kind of general aptitude test one would expect from a large enterprise's graduate scheme. What annoyed me further was that I hadn't even applied for the role - but had merely had an exploratory chat after the company's recruiter reached out. It was a rather disappointing experience and remains the only time I declined to return an assignment. 

**Machine Learning Engineer assignment tips TL;DR:**

>  ### Employer: 
>
> 
>
> 1. Respect the candidate and their time!!
> 2. Don't make assignment overly specific to your business problem
> 3. Test end-to-end productionisation skills, don't focus on Machine Learning skills
> 4. Reward candidate for their time by allowing them to push submission to their git portfolio



> ### Candidate:
>
> 1. Don't hesitate to ask for clarifications!
> 2. Presentation is key: keep the submission readable and concise!!
> 3. Provide an end-to-end solution: cover a bit of everything instead of going deep on one area (e.g. don't skip unit tests in favour of a more complex modelling pipeline)
> 4. Reproducility is key: virtual environments and Docker are your friends
> 5. A Jupyter Notebook won't cut it :)



In the final part 3, I will discuss interviewing and having realistic expectations.