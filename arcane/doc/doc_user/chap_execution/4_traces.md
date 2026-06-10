# Using Traces {#arcanedoc_execution_traces}

[TOC]

## Introduction {#arcanedoc_execution_traces_intro}

%Arcane provides a utility class (Arcane::TraceAccessor) for displaying traces
in modules. This class allows managing several types of traces: information,
errors, ...

If, in the module descriptor, the `parent-name` attribute of the `module`
element equals `Arcane::BasicModule` (the default), traces are automatically
available.

Traces are used like classic streams in C++, thanks to the << operator.

For example, to display an information trace:

```cpp
Arcane::TraceAccessor::info() << "Ceci est un message d'information";
```

All C++ types that have the `operator<<()` operator can be traced. For example:

```cpp
int z = 3;
Arcane::TraceAccessor::info() << "z vaut " << z;
```

Note that a carriage return is automatically performed between each message.
Consequently, adding a carriage return at the end of a trace causes a line
break.

## Trace Categories {#arcanedoc_execution_traces_class}

The trace methods are:
- \b Arcane::TraceAccessor::info() for information traces,
- \b Arcane::TraceAccessor::debug() for debug traces,
- \b Arcane::TraceAccessor::log() for log traces,
- \b Arcane::TraceAccessor::warning() for warning traces,
- \b Arcane::TraceAccessor::error() for error traces,
- \b Arcane::TraceAccessor::fatal() for fatal error traces, which stops
  execution. It is also possible to use the ARCANE_FATAL() macro to achieve the
  same behavior. The advantage of the macro is that it explicitly tells the
  compiler that an exception of type Arcane::FatalErrorException() is being
  thrown, which can help avoid compilation warnings.

Warning or error traces (Arcane::TraceAccessor::warning(),
Arcane::TraceAccessor::error(), and Arcane::TraceAccessor::fatal()) are always
displayed. For information (Arcane::TraceAccessor::info()) and debug
(Arcane::TraceAccessor::debug()) traces, the behavior depends on whether the
execution is sequential or parallel and whether ARCANE is compiled in debug or
optimized mode:
- in optimized mode, debug traces are never active. Furthermore, the `debug()`
  method is replaced by an empty method, meaning it does not consume any CPU
  resources.
- in optimized mode, by default, information traces are only displayed by
  subdomain 0. This behavior is configurable (see section
  \ref arcanedoc_execution_traces_config).
- in debug mode, traces from subdomain 0 are displayed on standard output.
  Traces from other subdomains are written to a file named 'output%n', where
  '%n' is the subdomain number.

Log traces are written to a file in the 'listing' directory, named 'log.%n',
where '%n' is the subdomain number.

There are 4 methods for parallel trace management:
- \b Arcane::TraceAccessor::pinfo() for information traces,
- \b Arcane::TraceAccessor::pwarning() for warning traces,
- \b Arcane::TraceAccessor::perror() for error traces,
- \b Arcane::TraceAccessor::pfatal() for fatal error traces, which stops
  execution.

For pinfo(), each subdomain displays the message. For the others
(Arcane::TraceAccessor::pwarning(), Arcane::TraceAccessor::perror(), and
Arcane::TraceAccessor::pfatal()), this means that each subdomain calls this
method (collective operation), and therefore only one trace will be displayed.
These parallel traces can be useful, for example, when you are certain that the
error will occur on all processors, such as an error in the dataset. You must
ensure that all subdomains call the collective methods, as otherwise it can lead
to code blocking.

It should be noted that if the Arcane::TraceAccessor::fatal() method is called
in parallel, the processes are generally terminated without warning. With
Arcane::TraceAccessor::pfatal(), it is possible to stop the code cleanly since
each subdomain generates the error.

There are three trace levels for the \c debug category: Arccore::Trace::Low,
\a Arccore::Trace::Medium, and Arccore::Trace::High. The default level is
\a Arccore::Trace::Medium.

```cpp
Arcane::TraceAccessor::debug(Arccore::Trace::Medium) << "Trace debug moyen"
Arcane::TraceAccessor::debug() << "Trace debug moyen"
Arcane::TraceAccessor::debug(Arccore::Trace::Low)    << "Trace debug affiché dès que le mode debug est utilisé"
```

## Trace Configuration {#arcanedoc_execution_traces_config}

It is possible to configure the desired debug level and the use of information
traces for each module in the ARCANE configuration file. This user configuration
file allows modifying the default behavior of certain architectural elements,
such as trace display. It is named <em>config.xml</em> and is located in
the <tt>.arcane</tt> directory of the user account running the execution.

Configuration is done using the \c name, \c info, and \c debug attributes of the
\c trace-module element. This element must be a child of the \c traces element.

- \b name specifies the name of the module concerned
- \b info equals \e true if information traces should be displayed, \e false
  otherwise.
- \b debug equals \e none, \e low, \e medium, or \e high according to the
  desired debug level. Debug traces at a level higher than requested are not
  displayed. The \e high level corresponds to all traces.

Here is an example file:

```xml
<?xml version="1.0" ?>
<arcane-config>
  <traces>
    <trace-class name="*" info="true" debug="none"/>
    <trace-class name="Hydro" info="true" debug="medium"/>
    <trace-class name="ParallelMng" info="true" print-class-name="false"
                 print-elapsed-time="true"/>
  </traces>
</arcane-config>
```

In the example, the user requests that information traces for all modules be
enabled by default, but not debug traces. For the Hydro module, information
traces and debug traces up to the \e medium level are displayed. For the
ParallelMng message class, the info and elapsed time are displayed, but not the
message class name (i.e., the beginning of the line '*I-ParallelMng'.

\note Regardless of the configuration, debug traces are not available in the
fully optimized version.

It is possible to dynamically change the information of a message class. For
example, the following code allows changing the verbosity level and displaying
the elapsed time from a module or service, but not the message class name:

```cpp
Arcane::ITraceMng* tm = traceMng();
Arcane::TraceClassConfig tcc = tm->classConfig("MyTest");
tcc.setFlags(Trace::PF_ElapsedTime|Trace::PF_NoClassName);
tcc.setVerboseLevel(4);
tm->setClassConfig("MyTest",tcc);
```


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_execution_env_variables
</span>
<span class="next_section_button">
\ref arcanedoc_execution_commandlineargs
</span>
</div>
