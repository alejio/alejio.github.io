---
title: "Engineering an FPL Brain: Why I'm Building in Public (While Losing)"
date: 2026-01-02T09:32:37Z
draft: false
---

Kicking off a series documenting the development of an ML system for Fantasy Premier League.

**Gameweek 19**. The exact midpoint of the Premier League season.


> After five months of work, 395 commits, and over 800 tests, I’ve built a production-minded system with reproducible pipelines, a CLI, and built-in observability. 

This isn't a success story. At least, not yet.

**tl;dr**

- **What I built**: A data pipeline, an ML model for expected points, a squad optimiser, and an experimental LLM agent for transfers.
- **Current performance**: Currently ranked ~2.7m out of 11m players (though I'm sitting +86 points above the average).
- **What’s novel**: Integrating betting-derived features, using a hauler-focused loss function, and prioritising agentic tools over pure optimisation.
- **What’s next**: Multi-gameweek planning, better uncertainty quantification, and further automation.


*Points per gameweek compared to FPL average and the current #1 manager:*

<img src="gameweek_points_timeline.png" alt="Gameweek Points Timeline" style="display: block; margin: 0 auto;" />

---

## What is FPL?

For the uninitiated: Fantasy Premier League is a free-to-play fantasy football game where you select 15 real Premier League players within a £100m budget. You earn points based on their real-world performances (goals, assists, clean sheets), make weekly transfers to optimise your squad, and compete against millions of other managers in a global leaderboard.

> The core problem: **predict player points and optimise transfers, team selection and captaincy under constraints**, then layer strategy over the optimiser.

For me, it's an excellent ML playground:

- **Bounded domain**: 15 players, £100m budget, clear rules
- **Weekly feedback loop**: no hiding from results
- **Rich data**: years of history, real-time updates, betting markets
- **Real competition**: 11M+ active players, measurable ranking

You can't hide behind vanity metrics when your rank is public. Every week, the system gets tested against millions of other decision-makers. The skin in the game is real.

---

## Why I'm doing this

I wanted a project where I could apply end-to-end data work without the restrictions a professional environment imposes: from data collection to ML to agentic systems, in a domain with clear, measurable outcomes.

Also, I got tired of consecutive years of sucking at FPL and decided to finally combine my professional specialty with one of my hobbies.

And despite all that sophistication? I'm currently ranked around 2.7 million out of 11 million managers 😅

Refreshingly, I'm +86 points above the average manager. But more interestingly, I think the system is finally converging to something good.

---

## The journey so far

The system now consists of 25 database tables and 15 domain services, with reproducible training, a growing test suite, and observability for the agent.

The architecture progressed through three distinct phases:

| Phase | Focus | Key Outcome |
|-------|-------|-------------|
| **Manual Control** | Interactive notebooks, rules-based picks | Learning the domain |
| **ML Predictions** | 156 features, custom loss functions | Better signals |
| **Agentic Reasoning** | LLM + tools, strategic context | Autonomous strategy |

The codebase lives across two repositories: `fpl-dataset-builder` handles data collection and enrichment, while `fpl-team-picker` handles analysis, ML, and the agent.

### Phase 1: Manual control (July–Aug)

I started where every data person starts: interactive notebooks. Specifically, to build a starting 15-player squad picker.

Using [Marimo](https://marimo.io/) (a reactive Python notebook), I built an interface for exploring the data and developing rules-based predictions of expected points. I could understand and tweak the predictions, and implemented a greedy optimiser to pick the best available player at each position under the budget constraints.

I got it to a place where the results made intuitive sense:

*First gameweek squad:*

<img src="first_squad.png" alt="First gameweek squad" style="display: block; margin: 0 auto;" />

I adapted the algorithm to pick the best transfers and starting 11 for each gameweek. But the problem with this rules-based approach was that it didn't allow self-improvement. The system wasn't learning, so I changed the paradigm.

### The data foundation

One thing I'm particularly proud of: the database.

*Database schema:*

<img src="database_schema.png" alt="Database schema" style="display: block; margin: 0 auto;" />

25 SQLite tables split into two layers:

**18 raw tables** capturing FPL data:
- 775 players, 380 fixtures, 38 gameweeks
- 13,395 player-gameweek observations
- My 270 picks across 18 gameweeks, chip usage, transfer history
- 190 fixtures with betting odds (Bet365, Pinnacle, market movements)

**7 derived tables** where the real value lives:
- Ownership trends: bandwagon detection, template players, transfer momentum
- Value analysis: points-per-million, buy/sell/hold recommendations
- Fixture difficulty: multi-factor analysis beyond FPL's simple 1–5 rating
- Team form: venue-specific attack/defence strength
- Betting features: implied probabilities, clean sheet likelihood, xG from odds

> **Pydantic validates everything.** This matters more than you'd think: code assistants love adding fallbacks to "make it work" rather than fixing upstream data issues. Strict validation forces correctness at the source.

### Phase 2: ML predictions (Sept–Dec)

With the data foundation in place, I built an ML pipeline with 156 features across categories: form, fixtures, ownership trends, betting odds, injury risk, venue-specific team strength.

Key design decisions:

- Custom loss functions focused on identifying **haulers** (players who score ~10+ points in a gameweek), not naively minimising average error
- Evaluation on >2 full gameweek holdout datasets 
- Hybrid training pipeline evaluating unified ML model vs position-specific models for GKP, DEF, MID, FWD (with hyperparameter tuning)
- A **simulated annealing (SA)** optimiser that explores beyond greedy local maxima

*Example of the transfer optimisation results:*

<img src="gameweek_manager_1.png" alt="Transfer optimisation" style="display: block; margin: 0 auto;" />

*Example of the starting 11 optimisation:*

<img src="gameweek_manager_2.png" alt="Starting 11 optimisation" style="display: block; margin: 0 auto;" />

But every gameweek still involved multiple "human-in-the-loop" decisions: model training strategy, risk appetite (conservative vs balanced vs aggressive), captain selection, etc. I was still making a lot of arbitrary calls.

### Phase 3: Agentic reasoning (Dec–today)

Predictions aren't enough. Consider this scenario: the ML model says Haaland is 8.2 **xP** (expected points), Saka is 7.8 xP. The optimiser picks Haaland.

But a strategic thinker might reason differently: there's a **double gameweek (DGW)** in 3 weeks, should I bank the transfer now? Or take a hit to avoid a price rise? What about **template safety** (minimising downside by covering highly-owned players) if I'm protecting rank?

This is strategic reasoning, not number crunching. The optimiser maximises expected points; it can't reason about context.

So I'm building an LLM agent. It uses the ML predictions as a tool, but adds a layer of strategic reasoning: DGWs, fixture swings, chip timing, template safety. It's experimental, but it's the "brain" in "Engineering an FPL Brain."

Pure mathematical optimisation is great for solving for a single moment, but it can be surprisingly fragile. An optimiser might suggest burning your last £0.5m in the bank to upgrade a bench defender today, only to leave you exactly £0.1m short of bringing back **Bryan Mbeumo** when he returns from **AFCON** in two weeks. That's where the agent comes in: it uses tools like `analyze_fixture_context` to spot upcoming fixture swings and `get_template_players` to ensure you aren't walking into a differential trap, providing the long-term strategic common sense that a simple calculator would miss.

*Running the agent CLI:*

<img src="transfer_agent_animated.gif" alt="Running the agent" style="width: 60%; display: block; margin: 0 auto;" />

What you're seeing: the orchestrator delegating to 5 specialised tools (multi-GW xP predictions, fixture context analysis, SA optimiser validation, squad weakness detection, template player analysis), then synthesising recommendations with strategic reasoning.

*Output of the agent CLI:*

<img src="agent_recommendation.png" alt="Agent output" style="width: 60%; display: block; margin: 0 auto;" />

At the moment it's restricted to transfer recommendations, rather than full team and captain selection. I've used [Pydantic AI](https://ai.pydantic.dev/) as the agentic framework for its structured outputs and [Logfire](https://logfire.pydantic.dev/) integration for observability.

*Logfire trace:*

<img src="logfire.png" alt="Logfire trace" style="width: 90%; display: block; margin: 0 auto;" />

---

## Where this is heading

**Short-term** (next 3–4 months): Refine the LLM agent: more tools, better reasoning, chip planning. Add uncertainty quantification to predictions. Improve captain accuracy (currently 22% hauler rate).

**Medium-term**: Multi-gameweek transfer planning (sequential optimisation). Benchmark against top 10K manager strategies. Automate more of the weekly workflow.

**The ambition**: a system that can reason about FPL at the level of a top player: not just crunch numbers, but understand when to take risks, when to play safe, when to go differential.

I don't know if I'll get there. But I'm building in (semi-)public, so you can follow along.

---

## What's next

This is the first in a series documenting the journey. Next posts will cover how the architecture evolved from notebooks to services, and when ML predictions aren't enough (the agent).

I'll share code, experiments that failed, and weekly accountability on how the system performs.

**Next: How the architecture evolved from notebooks to services** - coming soon!

---

### How to follow (and what feedback I’d love)

- **Follow**: connect on [LinkedIn](https://linkedin.com/in/alexspanos) (I’ll post each article + key charts there)
- **Tell me if I’m wrong**: if you’ve worked on probabilistic forecasting, ranking losses, or optimisation under uncertainty, I’d love pointers
- **Sanity-check strategy**: if you’re consistently top 10K and want to critique assumptions (risk, chip timing, “template” dynamics), DM me on LinkedIn