# Example n°4 {#arcanedoc_services_modules_simplecsvoutput_example4}

[TOC]

From this example, options or singletons will no longer be an issue.
Here, the focus will be on optimization, in case this service is used more
seriously than for simple debugging.

The result will therefore change slightly:
Results_Example4|Iteration 1|Iteration 2|Iteration 3|Somme
----------------|-----------|-----------|-----------|-----------
Nb de Fissions  |36         |0          |85         |121
Nb de Collisions|29         |84         |21         |134



## Initial entry point

Let's look at the `start-init` entry point:

`SimpleTableOutputExample4Module.cc`
\snippet SimpleTableOutputExample4Module.cc SimpleTableOutputExample4_init

In this example, we will create the rows and columns in the init instead of
doing it gradually.

Additionally, we will retrieve the row positions.

Indeed, internally, the value array is simply represented by an object of the
\arcane{RealUniqueArray2} class, a 2D array.
At the algorithmic level, this array is represented by a 1D array, with each row
placed side by side in memory.
Creating a row is therefore easy (if there is space) because you just need to
enlarge the 1D array.
But adding a column is noticeably more complex and time-consuming because you
are forced to shift the values of N-1 rows. And the more values there are in the
array, the longer it will take.

To avoid all this, we can create the rows and columns from the start.
Afterward, it will be enough to add the values where necessary.

To go even further, we save the positions of the two rows in class attributes.
This allows us to avoid performing a search for the `String` in the internal row
name array `StringUniqueArray`.
But this is dispensable given that the fewer rows there are, the less costly
this search is, and the more rows there are, the more complicated position
management becomes.
For this part, it's up to you.


## Loop entry point

Let's look at the `compute-loop` entry point:

`SimpleTableOutputExample4Module.cc`
\snippet SimpleTableOutputExample4Module.cc SimpleTableOutputExample4_loop

Here, we can see that the \arcane{ISimpleTableOutput::addElementInRow()} method
is used differently.
Indeed, this method is overloaded to allow the use of the row position instead
of its name, which avoids a String search.
Furthermore, this method returns a boolean that indicates whether the value
could be added or not, in case the position is incorrect.
(To simplify the example, I do not check the returned value).


## Exit entry point

Finally, let's look at the `exit` entry point:

`SimpleTableOutputExample4Module.cc`
\snippet SimpleTableOutputExample4Module.cc SimpleTableOutputExample4_exit

To vary things a bit and show how to use the values entered into the array, we
calculated the sum of the two rows and put these results into a new column,
`Somme`.

\note
We do not use an ArrayView in the service because it is impossible to create a
view on a column given that the column values are discontinuous.

This sum example shows that this service is not just a file writing service but
can also store values to exploit later.


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_services_modules_simplecsvoutput_example3
</span>
<span class="next_section_button">
\ref arcanedoc_services_modules_simplecsvoutput_example5
</span>
</div>
