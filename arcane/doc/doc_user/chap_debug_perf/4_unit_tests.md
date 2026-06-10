# Unit Test Support in %Arcane {#arcanedoc_debug_perf_unit_tests}

[TOC]

## Introduction {#arcanedoc_debug_perf_unit_tests_intro}

Unit tests in %Arcane are executed without running the simulation. They allow
you to test services, not modules. Their implementation is simple and fast:
services declare test methods in the descriptor ('.axl' file). These methods are
executed by a dedicated test module and time loop provided by %Arcane. To run
the tests, you simply need to add the service to the test module's list of
services.

This page describes the different steps required to build and run unit tests.
Note that, as of today, unit tests only function sequentially. Test report
generation is not yet implemented in parallel.

## Declaring Tests {#arcanedoc_debug_perf_unit_tests_decl}

Unit tests can be added to any service. To do this, you must declare the test
methods in the service's descriptor (the 'axl' file) in the following format:

```xml
<tests class-set-up="setUpForClass" test-set-up="setUp"
       class-tear-down="tearDownForClass" test-tear-down="tearDown">
  <test name="Test 1" method-name="myTestMethod1"/>
  <test name="Test 2" method-name="myTestMethod2"/>
  <test name="Test 3" method-name="myTestMethod3"/>
</tests> 
```

The descriptor above declares:

- 3 test methods: 'myTestMethod1', 'myTestMethod2', and 'myTestMethod3'. These 3
  tests have a name (the 'name' attribute) which is used for display during test
  execution.
- 2 optional methods: 'setUp' and 'tearDown', which are called respectively
  before and after each call to a test method. If no test methods are
  declared, 'setUp' and 'tearDown' are called n times. Note that it is entirely
  possible to declare only setUp or only tearDown.
- 2 optional methods: 'setUpForClass' and 'tearDownForClass', which are called
  respectively before and after the execution of all methods declared in the
  descriptor. Regardless of the number of methods declared, 'setUpForClass'
  and 'tearDownForClass' are called only once.

Once this work is done, you need to write the methods declared in the descriptor
in your service code. For the previous example, the '.h' file will look like
this:

```cpp
 ...
public:
 void setUpForClass();
 void tearDownForClass();
 void setUp();
 void tearDown();
 void myTestMethod1();
 void myTestMethod2();
 void myTestMethod3();
 ...
```

Nevertheless, if you forget to define one of the methods in your '.h', a
compilation error will occur. In the previous example, if you forget to define
'myTestMethod1', you will receive the following message:

```
error: no 'void MonServiceDeTest::myTestMethod1()' member function declared in class 'MonServiceDeTest'
```

## Assertions {#arcanedoc_debug_perf_unit_tests_assertions}

You must now code the test methods. Like most unit testing libraries (CppUnit,
GoogleTest...), %Arcane provides a set of assertions to test test results. These
assertions are available as C++ macros.

As of today, the available macros are:

- FAIL: allows a test to fail. This macro is useful, for example, to verify that
  an exception is thrown. You simply call FAIL after the instruction that should
  trigger the exception. If the exception is not triggered, the next instruction
  is executed and FAIL is called.
- ASSERT_TRUE(condition): checks that a boolean value is true. For example,
  ASSERT_TRUE(i<5).
- ASSERT_FALSE(condition): the inverse assertion of the previous one.
- ASSERT_EQUAL(expected, actual): checks for equality between the expected value
  for the test and the actual result obtained. This macro relies on a generic
  method (template) that uses the '==' operator. Therefore, this macro is valid
  for any type defining this operator. Note that Arcane's basic types meet this
  requirement (Integer, Real, Real2...).
- ASSERT_NEARLY_EQUAL(expected, actual): checks for 'nearly' equality between
  the expected value for the test and the actual result obtained. This assertion
  is useful for testing equality between reals despite machine inaccuracies.
  This macro relies on a generic method (template) that uses the
  'math::isNearlyEqual' method. Therefore, this macro is valid for any type
  defining this method. This is the case for Arcane reals.
- ASSERT_NEARLY_EQUAL_EPSILON(expected, actual, epsilon): functions like the
  previous assertion but uses a comparison epsilon provided as a parameter by
  the caller.

Here are some examples of how to use these macros:

```cpp
ASSERT_TRUE(i <= 5);
ASSERT_FALSE(i > 5);
ASSERT_EQUAL(5, x);
ASSERT_NEARLY_EQUAL(5.5, y);
```

### Parallel Unit Tests {#arcanedoc_debug_perf_unit_tests_parallel}

Since version 2.20 of %Arcane, it is possible to use unit tests in parallel. To
do this, a version of the macros is available by specifying an instance of
Arcane::IParallelMng as a parameter. These macros are semantically identical to
the sequential version and are prefixed with `PARALLEL_`, such as
PARALLEL_ASSERT_TRUE or PARALLEL_ASSERT_NEARLY_EQUAL. These calls are
collective, and the test is considered failed if any thread fails.

Here is an example of use:

```cpp
using namespace Arcane;
IParallelMng* pm = ...;
Real deltat = ...;
PARALLEL_ASSERT_NEARLY_EQUAL(deltat,1.0,pm);
```

### Special Case of Exceptions {#arcanedoc_debug_perf_unit_tests_exception}

Sometimes, you want to develop a unit test to verify that a method throws an
exception. Suppose you want to run a test that throws MyException in the
myMethod() method. The technique is to call the FAIL macro immediately after
myMethod(). Thus, if the exception is thrown, the macro is not called and the
test passes. You then simply need to handle the exception in a catch block.

```cpp
try {
  myMethod();   // must throw my exception...
  FAIL;         // ... if I am here, the exception was not thrown
} catch (const MyException& e) {
  // ok, the exception was thrown
}
```

Sometimes the method does not raise an exception but calls the
TraceAccessor::fatal() method of Arcane's trace manager (see
\ref arcanedoc_execution_traces).
This method raises an exception of type Arcane::FatalErrorException, which
should be handled like in the example above.

## The Data File {#arcanedoc_debug_perf_unit_tests_data}

Unit tests run using a specific service and time loop provided by %Arcane. The
only thing you need to do is select the time loop in question in the code's data
file and add your test service to the module's list of services.

The example below shows a typical data file with 'MonServiceDeTest' in the test
services list.

```xml
<arcane>
  <titre>My Test Case</titre>
  <description>Description of my test case</description>
  <boucle-en-temps>UnitTest</boucle-en-temps>
</arcane>
 ...
<module-test-unitaire>
  ...
  <xml-test name="MonServiceDeTest">
    <!-- here are the data for my test service (if it has any!)... -->
  </xml-test>
  ...
</module-test-unitaire>
```

## Execution {#arcanedoc_debug_perf_unit_tests_run}

You then simply run your program as usual, providing the file defined in the
previous step as the dataset. The test module will then perform an iteration by
triggering all the test methods of the services provided in the data file. You
will then be able to see the trace of the test execution in the listing, as
shown below.

It should be noted that the listing also indicates the path to the test report.

```
...
*I-Master     *** ITERATION        1  TEMPS 1.000000000000000e+00  BOUCLE        1  DELTAT 1.000000000000000e+00 ***
*I-Master     Date: 2013-05-28T14:52:28 Conso=(R=0,I=0,C=0) Mem=(62,m=62:0,M=62:0,avg=62)
*I-UnitTest   [OK   ] myTestMethod1
*I-UnitTest   [ECHEC] myTestMethod2 (line 136 in virtual void MonServiceDeTest::myTestMethod2())
*I-UnitTest           Obtenu : 6.5. Attendu : 5.5.
*I-UnitTest   [OK   ] myTestMethod3
...
*I-UnitTest   Sortie du rapport de test unitaire dans '/tmp/moncas/output/listing/unittests.xml'
...
```

The listing is an XML file. It has the following format:

```xml
<unit-tests-results>
  <service name="MonServiceDeTest">
    <unit-test method-name="Test 1" name="myTestMethod1" result="success"/>
    <unit-test method-name="Test 2" name="myTestMethod2" result="failure">
      <exception file="/tmp/monprojet/src/MonServiceDeTest.cc"
                 line="136" message="Obtained: 6.5. Expected: 5.5."
                 where="virtual void MonServiceDeTest::myTestMethod2()"/>
    </unit-test>
    <unit-test method-name="Test 3" name="myTestMethod3" result="success"/>
  </service>
</unit-tests-results>
```

## Using Your Own Time Loop {#arcanedoc_debug_perf_unit_tests_own_timeloop}

Sometimes, before running unit tests, you might want to perform initializations
that are done in the application module entry points. However, since unit tests
are executed within a specific time loop, these entry points are not called.

To bypass this, it is possible to create your own time loop. This time loop must
necessarily include the following 3 entry points:

- UnitTest.UnitTestInit in the Init section,
- UnitTest.UnitTestDoTest in the ComputeLoop section,
- UnitTest.UnitTestExit in the Exit section.

You must remember to add the 'UnitTest' module to the loop's list of modules.

You simply need to insert your own initialization entry points before
'UnitTest.UnitTestInit'. The 'ComputeLoop' section usually only contains the
'UnitTest.UnitTestDoTest' entry point.

```xml
<arcane-config code-name="MyCode">
  <time-loops>
    <time-loop name="MyTimeLoop">
      <title>My nice timeloop</title>
      <modules>
        <module name="UnitTest" need="required"/>
        ...
      </modules>

      <entry-points where="init">
        ...
        <entry-point name="UnitTest.UnitTestInit"/>
      </entry-points>

      <entry-points where="compute-loop">
        <entry-point name="UnitTest.UnitTestDoTest"/>
      </entry-points>

      <entry-points where="exit">
        <entry-point name="UnitTest.UnitTestExit"/>
      </entry-points>
    </time-loop>
  </time-loops>
</arcane-config>
```

Note that you must not forget to change the name of the time loop in the data
file!


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_debug_perf_compare_bittobit
</span>
<span class="next_section_button">
\ref arcanedoc_debug_perf_cupti
</span>
</div>
