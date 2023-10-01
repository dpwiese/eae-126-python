# EAE 126 Computational Aerodynamics

This repository contains code from the UC Davis course EAE 126 Computational Aerodyamics from the Spring 2011 semester.
The code was originally written in Matlab and converted to Python to share in this repository.

This repository exists only because I was interested in investigating how to best convert Matlab code to Python - the solution I chose was [ChatGPT](http://chat.openai.com).
I intend to write a bit more about the conversion process, but as far as this repository is concerned there are a few things to note.
The first is that for the most part, the [code smells](https://wiki.c2.com/?CodeSmell) which exist in the produced code were due to their existence in the original code.
Now, to some extent the original code was written to best map to the hand calculations of the underlying equations, but beyond that it was not particulary well written more than a decade ago when I was in college.
As bad as it was, I made minimal attempts to refactor the code prior to conversion, and minimal manual massaging of the code after conversion.
As a result, the code contain in this repository certainly contains bugs, and leaves ample room for improvement.

Nevertheless, I'm happy to have gone through the conversion exercise for two reasons: now I, and anyone else without a Matlab license can run this code with available open-source software, and secondly, I better understand and can report the use of ChatGPT as a very useful tool to convert subsequent codes from Matlab and hopefully enable others to do the same.

* [Project 1: Steady, inviscid, adiabatic, incompressible, and irrotational 2D flows over cylinder](https://github.com/dpwiese/eae-126-python/tree/main/project1)
* [Project 2: Joukowski transformation and airfoils](https://github.com/dpwiese/eae-126-python/tree/main/project2)
* [Project 3: Small disturbance theory for airfoils and bodies of revolution](https://github.com/dpwiese/eae-126-python/tree/main/project3)
* [Project 4: High and Low Aspect Ratio Wings](https://github.com/dpwiese/eae-126-python/tree/main/project4)
* [Project 5: Steady, inviscid, adiabatic, compressible, and irrotational flows over airfoils - numerical solutions to thickness problem](https://github.com/dpwiese/eae-126-python/tree/main/project5)
* [Project 6: Steady, inviscid, adiabatic, compressible, and irrotational flows over airfoils - numerical solutions to lifting Problem](https://github.com/dpwiese/eae-126-python/tree/main/project6)
* [Project 7: Steady, inviscid, adiabatic, compressible, and irrotational 2D flows over airfoils - numerical solutions: supersonic](https://github.com/dpwiese/eae-126-python/tree/main/project7)
* [Project 8: Transonic Flow and Boundary Layers](https://github.com/dpwiese/eae-126-python/tree/main/project8)
