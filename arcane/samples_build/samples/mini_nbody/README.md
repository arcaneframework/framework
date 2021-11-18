## Usage

En séquentiel avec 1000 corps (le défaut)
~~~
MiniNbody
~~~


8 threads et 500 corps
~~~
MiniNbody -A,T=8 -A,NbBody=500
~~~

Sur accélérateur NVDIA (si Arcane est compilé avec le support correspondant)

~~~
MiniNbody -A,AcceleratorRuntime=cuda
~~~

~~~
MiniNbody -A,AcceleratorRuntime=cuda -A,NbBody=4000
~~~
