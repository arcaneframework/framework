# Exemple n°5 {#arcanedoc_services_modules_simplecsvoutput_example5}

[TOC]

In this example, we will show an example of using table reading/filling via the
position of a pointer.

Indeed, until now, we have filled our tables of values with line/column names,
placing the values next to each other.
It turns out that internally, when an element is added or modified, there is a
pointer that updates and points to the last manipulated element.

\remark
Pointer in the sense of "pointer to a position in the 2D array," not in the
sense of "C pointer to a memory area."

We can therefore use this pointer to modify values around the last manipulated
value.

| Results_Example5             | Iteration 1 | Iteration 2 | Iteration 3 | Somme |
|------------------------------|-------------|-------------|-------------|-------|
| Nb de Fissions               | 36          | 0           | 85          | 121   |
| Nb de Fissions (div par 2)   | 18          | 0           | 42.5        | 60.5  |
| Nb de Collisions             | 29          | 84          | 21          | 134   |
| Nb de Collisions (div par 2) | 14.5        | 42          | 10.5        | 67    |



## Initial entry point

Let's look at the `start-init` entry point:

`SimpleTableOutputExample5Module.cc`
\snippet SimpleTableOutputExample5Module.cc SimpleTableOutputExample5_init

Here, nothing particularly original, apart from the two extra lines:
`(div by 2)`.
These two lines will contain the number of fissions/collisions in an iteration
divided by 2.



## Loop entry point

Let's look at the `compute-loop` entry point:

`SimpleTableOutputExample5Module.cc`
\snippet SimpleTableOutputExample5Module.cc SimpleTableOutputExample5_loop

A new method appears: \arcane{ISimpleTableOutput::editElementDown()}.
This method allows modifying the "cell" below the "cell" that was just modified.
Let's take the two lines dedicated to fission. The first line adds a value to
the `Nb de Fissions` line.
Internally, a pointer is modified and now points to the value that was just
added.
The second line calls the new method. This method will take the pointer, search
for the "cell" below, and replace its value with `nb_fissions/2`. By default,
the pointer will then be updated and point to this manipulated "cell." So if we
wanted to add, for example, `nb_fissions*2` right below, we could make another
call to the \arcane{ISimpleTableOutput::editElementDown()} method right after.

In the case where there is a modification of `nb_fissions` between the two
lines (without touching the table):

```cpp
options()->csvOutput()->addElementInRow(pos_fis, nb_fissions);
nb_fissions += 456;
options()->csvOutput()->editElementDown(nb_fissions/2.); // Pas la bonne valeur !!!
```
We could do this:

```cpp
options()->csvOutput()->addElementInRow(pos_fis, nb_fissions);
nb_fissions += 456;
options()->csvOutput()->editElementDown(element()/2.); // C'est correct !!!
```

\arcane{ISimpleTableOutput::element()} is a method that allows retrieving the
value of the "cell" pointed to by the pointer. This can be practical in this
case, for example (we agree that if we don't use the pointer and the associated
methods, it's a useless method).

\note
There is not only the \arcane{ISimpleTableOutput::editElementDown()} method;
there are equivalent methods for the four directions:
\arcane{ISimpleTableOutput::editElementUp()}
\arcane{ISimpleTableOutput::editElementLeft()}
\arcane{ISimpleTableOutput::editElementRight()}
Same for \arcane{ISimpleTableOutput::element()}.


## Exit entry point

Finally, let's look at the `exit` entry point:

`SimpleTableOutputExample5Module.cc`
\snippet SimpleTableOutputExample5Module.cc SimpleTableOutputExample5_exit

This entry point is identical to that of the previous example.


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_services_modules_simplecsvoutput_example4
</span>
<span class="next_section_button">
\ref arcanedoc_services_modules_simplecsvoutput_example6
</span>
</div>
