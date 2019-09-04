---
layout: post
title: Hiring Machine Learning Engineers Part 1 - Context
categories: [Data Science]
---

Recently, my team decided to hire for a *Machine Learning Engineer*. Whilst considering the requirements for this role, a relatively new and ill-defined one in the industry, I found myself thinking about its contrast with the *Data Scientist* - incidentally, the role I continue to identify mostly with.

In the recent past this kind of distinction only made academic sense to me. I will confess to subconsciously thinking that a Machine Learning Engineer is someone who took the [fast.ai](https://www.fast.ai/) course and now builds Deep Learning models on GPU's, claiming out-of-the-world performances, without considering caveats and biases of the underlying datasets.

But the real differences only really clicked after having experienced first-hand what it takes to set up an autonomous data and ML-enabled product team.

> Alex: "How to name the role is superficial. In the end of the day, Machine Learning Engineers or Data Scientists simply build the models and work with the team's Software Engineers to integrate all those fancy models in production, right?"
>
> Software Engineers in the room: *...produce Mona Lisa-esque smile.. awkward silence ensues...*]

The hiring process itself, reviewing applications, assignment submissions and conducting video/onsite interviews with my Data Science/ML peers was an awesome experience, so I'd like to share some of my thoughts and observations, in the hope that they may prove useful for people on either side of the process.

The story is broken down into 2 parts:

1. Context
2. Defining a hiring process.

## Context

Before jumping onto describing the Machine Learning Engineer, it is useful to remember the evolution of its sibling, the Data Scientist.

Nowadays, the unicorn halo crowning those Data Scientists who specialise in narrow scientific methodologies has generally fallen away. Typically, the most effective Data Scientist profiles in the industry, in most cases, combine scientific rigour and a product/business-first mentality - and rather crucially a strong will to become a "good programmer".

Hence, the emergence of the Third Wave Data Scientist (link at the), following its predecessor the *proof-of-concept* Data Scientist, and the ancestral Data Analyst variant. Having experienced these transitions first hand, it now feels like the industry, at least in its major hubs, is learning to put Machine Learning practitioners to use much more effectively.

This is part of an overall tech ecosystem evolution fuelled by ever-increasing venture capital and the ensuing explosive birth rate of startups requiring "jacks-of-all trades"- and is closely linked to the proliferation of the Machine Learning Engineer, following that of the Data Engineer.

As tech history teaches us, nomenclature is transient; at the moment it means different things at different places. But today what really matters is that it doesn't make sense to place hard borders between roles and teams. I won't go into too much detail here, as a bunch of amazing tech blogs have already covered this extensively.

So you will be spared the cringe of another Venn Diagram; it suffices to say that the set of common skills across all areas is becoming increasingly larger over time. A modern ML-powered team requires flexibility and interchangeability of duties.

> *On a side note I can't resist: the football nerd in me sees a strong parable with Total Football, a revolutionary football system born in Holland in the 1960s. In Total Football, no outfield player had a fully deterministic role; any player could attack an area traditionally "owned" by another teammate, who would in turn have to cover the area exposed by the attacker. As a style of play, it was simultaneously incredibly stylish but also very successful in terms of gaining a competitive advantage; I would also assume that it made footballers enjoy their job more!*
>
> **Sketch** : **Total Football intuition** (taken from Imperial BS Student Blog, where someone had similar thoughts with me):
>
> ![total football](http://i.imgur.com/W5LzxMi.png)

But as mentioned above, whilst duties of ML-powered team members overlap significantly, there are important differences, particularly between the desirable **mentalities** of Machine Learning Engineers and Data Scientists.

A Machine Learning *Engineer* is **a doer**, essentially a Software Engineer in disguise. They are able to implement the end-to-end workflow required for deploying an ML service within an organisations' existing tech infrastructure.

Speaking the DevOps and Backend Engineering languages, they are the ones who make the work of model-building Data Scientists possible and in comparison their technical responsibilities are much broader and more fundamental. When not deploying models to production, you will find them building data pipelines, improving monitoring capabilities and tinkering with nascent Machine Learning libraries in exotic programming languages. Whilst constantly fixing bugs across the stack.

In contrast, a Data *Scientist* is primarily **a thinker**, whose main task is coming up with data-driven ways of solving a business/product problem - and then actually solving it. They tend to spend more time with the product side of organisations and eventually learn to condense high-level "business" talk into a set of analytical tasks. 

> Data *Scientists* exercise failing; Machine Learning *Engineers* exercise **not** failing.

The clue is in the name.

Failing productively is a major skill for Data Scientists to develop, as their days are spent eliminating paths defined by their experimentation plan. In contrast, Machine Learning Engineers dream of their end-to-end production workflow working faultlessly like clockwork; to them failure is abhorrent!

## So, whom to hire?

Businesses look to Data Science to make data useful for them - whom to hire depends on the maturity of the business' data infrastructure.

Bringing in Machine Learning Engineers early to work with Software Engineers has multiple benefits, as

1. They can participate and outline Data Science-related requirements to Engineers building out data infrastructure and pipelines; and
2. they are capable of building end-to-end prototype workflows, thus setting the ground for Data Scientists to become quickly productive.

But if the business has not ironed out its data infrastructure,  Data Scientists are not the right hire in the short term. Data is oxygen for them, and its absence is detrimental for both job satisfaction and productivity. In the longer term, the infusion of a scientific mindset and the value-add provided by Data Scientists can become a major asset and a differentiator for businesses. But in most cases that comes later in the journey: in the meanwhile, avoid becoming a trophy Data Scientist!

## Next post preview

In Part 2, I will cover how to define a hiring process for a Machine Learning Engineer:

- Articulating role responsibilities;
- using a take-home task as a means of mutual education and communication; and
- having realistic expectations.

## Resources

1. [The Third Wave Data Scientist](https://towardsdatascience.com/the-third-wave-data-scientist-1421df7433c9)
2. [Data Engineers](https://medium.com/@rchang/a-beginners-guide-to-data-engineering-part-i-4227c5c457d7)
3. [Machine Learning faster](http://nlathia.github.io/2019/08/13/Machine-learning-faster/)
4. [Can ‘Total Football’ work in organisations?](https://www.imperial.ac.uk/business-school/intelligence/student-blog/can-total-football-work-organisations/)
5. [But what is this Machine Learning Engineer actually doing?](https://medium.com/@tomaszdudek/but-what-is-this-machine-learning-engineer-actually-doing-18464d5c699)
6. [Avoiding being a trophy Data Scientist](https://peadarcoyle.com/2017/07/23/avoiding-being-a-trophy-data-scientist)