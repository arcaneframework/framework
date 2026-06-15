# Main File {#arcanedoc_examples_simple_example_main}

[TOC]

Now, let's look at the `main.cc` file. This file contains the `main()` function,
which will be launched when HelloWorld starts. Here is what it looks like:

## main.cc {#arcanedoc_examples_simple_example_main_maincc}
```cpp
// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#include <arcane/launcher/ArcaneLauncher.h>

using namespace Arcane;

int
main(int argc,char* argv[])
{
  ArcaneLauncher::init(CommandLineArguments(&argc,&argv));
  auto& app_build_info = ArcaneLauncher::applicationBuildInfo();
  app_build_info.setCodeName("HelloWorld");
  app_build_info.setCodeVersion(VersionInfo(1,0,0));
  return ArcaneLauncher::run();
}
```
The `main()` function is used to launch %Arcane and our application.
Therefore, this function will practically never be modified (except to update
the code version). In more advanced cases, we must modify `main()` to,
for example, change the memory allocator that will be used in %Arcane for
our application.
We can also use this function to run calculations without modules,
by using the %Arcane classes.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_examples_simple_example_config
</span>
<span class="next_section_button">
\ref arcanedoc_examples_simple_example_cmake
</span>
</div>
