# RNG Service {#arcanedoc_examples_concret_example_rng}

[TOC]

In this subsection, we will not talk about modules
since most of what needed to be said
was covered in subsection \ref arcanedoc_examples_simple_example.

Here, we will talk about services, and more specifically the RNG service. We
will not talk about the implementation of the different interface methods; that
is not the purpose of this subsection.
Therefore, the file `RNGService.cc` will be intentionally omitted.

## IRandomNumberGenerator.h {#arcanedoc_examples_concret_example_rng_irandomnumbergeneratorh}

This is the interface used by this service:
```cpp
// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IRandomNumberGenerator.h                                    (C) 2000-2022 */
/*                                                                           */
/* Interface for random number generator.                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_IRANDOMNUMBERGENERATOR_H
#define ARCANE_IRANDOMNUMBERGENERATOR_H

#include "arcane/utils/Array.h"
#include "arcane/utils/UtilsTypes.h"

namespace Arcane
{

class ARCANE_CORE_EXPORT IRandomNumberGenerator
{
 public:
  virtual ~IRandomNumberGenerator() = default;

 public:
  virtual bool initSeed() = 0;
  virtual bool initSeed(ByteArrayView seed) = 0;
  virtual ByteConstArrayView viewSeed() = 0;
  virtual ByteUniqueArray emptySeed() = 0;
  virtual Integer neededSizeOfSeed() = 0;
  virtual bool isLeapSeedSupported() = 0;
  virtual ByteUniqueArray generateRandomSeed(Integer leap = 0) = 0;
  virtual ByteUniqueArray generateRandomSeed(ByteArrayView parent_seed, Integer leap = 0) = 0;
  virtual bool isLeapNumberSupported() = 0;
  virtual Real generateRandomNumber(Integer leap = 0) = 0;
  virtual Real generateRandomNumber(ByteArrayView seed, Integer leap = 0) = 0;
};

}

#endif

```
There isn't much to say here; it is a fairly classic interface. We have the
first lines with the license and a short description of the interface (lines
1-12).
Then, we have a class with virtual methods equal to 0, which makes them pure
virtual methods that must be implemented in an implementation.

Note however that the seeds are represented by byte arrays (ByteUniqueArray)
and are manipulated by views (ByteArrayView). This allows the application to be
independent of the implementation of `Arcane::IRandomNumberGenerator`.

Indeed, the implementation in our mini-app uses 64-bit seeds (i.e., 8 bytes),
but another implementation might use 32-bit seeds. Therefore, we use arrays
whose sizes are defined at runtime (the choice of implementation is made in the
dataset, making it impossible to know the size of a seed at compilation).

## RNG.axl {#arcanedoc_examples_concret_example_rng_rngaxl}

Now let's look at the `.axl`:
```xml
<?xml version="1.0"?>
<service name="RNG" version="1.0" singleton="true">

  <description>RNG service dataset</description>
  <interface name="IRandomNumberGenerator"/>

  <variables>
  </variables>

  <options>
  </options>
</service>
```
We can see that it strongly resembles a module's `.axl`.
```xml
<service name="RNG" version="1.0" singleton="true">
```
Here, there is a new feature: `singleton="true"`. This part tells us that it is
possible to use this service as a singleton.

____

```xml
<interface name="IRandomNumberGenerator"/>
```
We give %Arcane the service interface.
Indeed, as we will see later, the `RNGService` class does not directly inherit
from the `Arcane::IRandomNumberGenerator` interface. This is handled by the
generated file `RNG_axl.h`.

## RNGService.h {#arcanedoc_examples_concret_example_rng_rngserviceh}

Here is the RNG service header:

```cpp
// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RNGService.hh                                               (C) 2000-2022 */
/*                                                                           */
/* Implementation of a random number generator.                              */
/* Based on the Quicksilver generator (LLNL).                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <arcane/core/IRandomNumberGenerator.h>
#include "rng/RNG_axl.h"

using namespace Arcane;

class RNGService
: public ArcaneRNGObject
{
public:
  RNGService(const ServiceBuildInfo & sbi)
    : ArcaneRNGObject(sbi)
    , m_seed(0)
    {}
  
  virtual ~RNGService() {};

public:
  bool initSeed() override;
  bool initSeed(ByteArrayView seed) override;

  ByteUniqueArray emptySeed() override;
  ByteConstArrayView viewSeed() override;

  Integer neededSizeOfSeed() override;

  bool isLeapSeedSupported() override { return false; };
  ByteUniqueArray generateRandomSeed(Integer leap = 0) override;
  ByteUniqueArray generateRandomSeed(ByteArrayView parent_seed, Integer leap = 0) override;

  bool isLeapNumberSupported() override { return false; };
  Real generateRandomNumber(Integer leap) override;
  Real generateRandomNumber(ByteArrayView seed, Integer leap = 0) override;

protected:
  Real _rngSample(Int64* seed);
  void _breakupUInt64(uint64_t uint64_in, uint32_t& front_bits, uint32_t& back_bits);
  uint64_t _reconstructUInt64(uint32_t front_bits, uint32_t back_bits);
  void _pseudoDES(uint32_t& lword, uint32_t& irword);
  uint64_t _hashState(uint64_t initial_number);

protected:
  Int64 m_seed;
  const Integer m_size_of_seed = sizeof(Int64);
};

ARCANE_REGISTER_SERVICE_RNG(RNG, RNGService);
```
This is also very similar to a module.
The biggest difference is at the level of the `override` methods.
Indeed, in a module, we override the entry points that we defined in the `.axl`.
Here, in this service, we override the virtual methods of our interface.
Services cannot have entry points.

\note Regarding the `includes`, you must ensure that the interface declared in
the `.axl` is accessible by the `.h` generated from the `.axl`. For example,
here, the order of the `includes` is very important.
In the `.axl`, we declare this:

```xml
<interface name="IRandomNumberGenerator"/>
```
In the file `RNG_axl.h` that will be generated, we will obtain this:
```cpp
//[...]
class ArcaneRNGObject
  : public Arcane::BasicService
  , public Arcane::IRandomNumberGenerator
{
//[...]
```
Therefore, we must include the interface before this file:
```cpp
#include <arcane/core/IRandomNumberGenerator.h>
#include "rng/RNG_axl.h"
```



____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_examples_concret_example_config
</span>
<span class="next_section_button">
\ref arcanedoc_examples_concret_example_build
</span>
</div>
