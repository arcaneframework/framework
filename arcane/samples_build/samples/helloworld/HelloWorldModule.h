// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef HELLOWORLDMODULE_H
#define HELLOWORLDMODULE_H

#include "HelloWorld_axl.h"

using namespace Arcane;

/**
 * Représente un module trés basique ne contenant aucune variable et un unique 
 * point d'entrée permettant d'afficher la chaine de caractères "Hello World!".
 */
class HelloWorldModule
: public ArcaneHelloWorldObject
{
 public:
  /** Constructeur de la classe */
  HelloWorldModule(const ModuleBuildInfo & mbi) 
  : ArcaneHelloWorldObject(mbi) { }

  /** Destructeur de la classe */
  ~HelloWorldModule() { }

 public:
  /**
   * Affiche la chaine de caract?res "Hello World!". 
   * Cette méthode est un point d'entrée du module enregistré sous le nom
   * \c PrintHelloWorld.
   */
  void printHelloWorld() override;

  /** Retourne le numéro de version du module */
  VersionInfo versionInfo() const override { return VersionInfo(1, 0, 0); }
};

#endif
