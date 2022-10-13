# Service RNG {#arcanedoc_examples_concret_example_rng}

[TOC]

Dans ce sous-chapitre, nous n'allons pas parler des modules
puisque la majeur partie de ce qu'il y avait à dire
a été abordé dans le sous-chapitre \ref arcanedoc_examples_simple_example.

Ici, nous allons parler service, et plus précisément du service
RNG. Nous n'allons pas parler de l'implémentation des différentes
méthodes de l'interface, ce n'est pas le but de ce sous-chapitre.
Donc, le fichier `RNGService.cc` sera volontairement omis.

## IRandomNumberGenerator.h {#arcanedoc_examples_concret_example_rng_irandomnumbergeneratorh}

Voici l'interface utilisé par ce service :
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
/* Interface pour générateur de nombres aléatoires.                          */
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
Il n'y a pas grand chose à dire ici, c'est une interface assez
classique. On a les premières lignes avec la licence et
une courte description de l'interface (lignes 1-12).
Ensuite, on a une classe avec des méthodes virtuelles égales à 0, ce qui
en fait des méthodes virtuelles pures que l'on doit implémenter dans une
implémentation.

A noter cependant que les graines sont représentées par des tableaux de Bytes
(ByteUniqueArray) et sont manipulées par des vues (ByteArrayView).
Cela permet de rendre l'application indépendante de l'implémentation de
`Arcane::IRandomNumberGenerator`.

En effet, l'implémentation présente dans notre mini-app utilise des graines
sur 64 bits (donc 8 octets) mais il se pourrait qu'une autre implémentation
utilise des graines sur 32 bits. Donc on utilise des tableaux de tailles
définis lors de l'exécution (le choix de l'implémentation étant faite dans
le jeu de données, impossible de connaitre la taille d'une graine à la compilation).

## RNG.axl {#arcanedoc_examples_concret_example_rng_rngaxl}
Voyons maintenant le `.axl` :
```xml
<?xml version="1.0"?>
<service name="RNG" version="1.0" singleton="true">

  <description>Jeu de données du service RNG</description>
  <interface name="IRandomNumberGenerator" />

  <variables>
  </variables>

  <options>
  </options>
</service>
```
On peut voir que ça ressemble fortement à un `.axl` de module.
```xml
<service name="RNG" version="1.0" singleton="true">
```
Ici, on a une nouveauté : `singleton="true"`. Cette partie
nous dit qu'il est possible d'utiliser ce service en tant que
singleton.

____

```xml
<interface name="IRandomNumberGenerator" />
```
On donne à %Arcane l'interface du service.
En effet, comme on le verra après, la classe `RNGService` n'hérite pas
directement de l'interface `Arcane::IRandomNumberGenerator`. C'est géré par le
fichier généré `RNG_axl.h`.

## RNGService.h {#arcanedoc_examples_concret_example_rng_rngserviceh}

Voici le header du service RNG :
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
/* Implémentation d'un générateur de nombres aléatoires.                     */
/* Basé sur le générateur de Quicksilver (LLNL).                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <arcane/IRandomNumberGenerator.h>
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
Là aussi, c'est très semblable à un module.
La plus grosse différence est au niveau des méthodes override.
En effet, dans un module, on override les points d'entrées que l'on a 
définit dans le `.axl`. Ici, dans ce service, on override les méthodes
virtuelles de notre interface.
Les services ne peuvent pas avoir de points d'entrées.

\note Au niveau des `includes`, il faut s'assurer que l'interface
déclarée dans le `.axl` soit accessible par le `.h` généré à partir du
`.axl`. Par exemple, ici, l'ordre des `includes` est très important.
Dans le `.axl`, on déclare ceci :
```xml
<interface name="IRandomNumberGenerator" />
```
Dans le fichier `RNG_axl.h` qui sera généré, on obtiendra cela :
```cpp
//[...]
class ArcaneRNGObject
  : public Arcane::BasicService
  , public Arcane::IRandomNumberGenerator
{
//[...]
```
Il faut donc inclure l'interface avant ce fichier :
```cpp
#include <arcane/IRandomNumberGenerator.h>
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