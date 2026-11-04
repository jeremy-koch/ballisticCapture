# Spec history

The specs and prompts for this project as they changed, oldest first. Each entry is the prompt as sent, with the date and the model.

To be honest I did not iterate much on this, I was more concerned with getting you an example. 

## 2026-11-03: the original spec (Claude Opus 5.5, Claude Code)


> I'd like Python code that compares a Hohmann-style transfer and a ballistic transfer from Earth to the Moon, in terms of total Δv and travel time.
>
> Model: Planar bicircular model: Earth at the origin of an Earth-centered frame, the Moon and the Sun each on prescribed circular orbits in the same plane. The spacecraft feels Earth, Moon, and Sun gravity. Include the indirect terms (the Sun's and Moon's acceleration of Earth), since the frame is centered on Earth. Use km and s.
>
> Mission. Start in a 200 km circular low Earth orbit, with a single tangential departure burn. The goal is a 100 km circular lunar orbit. Capture Δv = speed relative to the Moon at closest approach minus circular speed at 100 km altitude. Count a trajectory as "captured" if its two-body energy relative to the Moon stays negative for at least one lunar orbit. Log departure Δv, capture Δv, total Δv, and time of flight for every run.
>
> Numerics. scipy.integrate.solve_ivp with tight tolerances.
>
> Plots. (a) The best ballistic and the Hohmann trajectories. (b) A capture map: departure speed vs. phase, colored by capture Δv. (c) Total Δv vs. time of flight for all successful runs.
>
> Please propose a plan and file structure before writing code. Keep the physics in small, separate functions that I can test by hand.

Comment: the code was impressively complicated and I didn't have time to dig-in to all the details. So I gave a second prompt.

## 2026-11-04: simplifying code (Claude Opus 5.5, Claude Code)

> I want to give this repository as an example for my first-year programming course. Right now, it seems a bit overwhelming and might encourage "black box" use of AI. Could we simplify it a bit, even if it means we lose some features?
> 
> Is this basically a shooting method problem, but in 2D? I think we should keep the time-stepping and sweeping aspects, but maybe ditch the other stuff. I bet we can get something that is 80% as good with a lot simpler code.
