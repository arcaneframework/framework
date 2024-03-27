// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SodMeshGenerator.h                                          (C) 2000-2024 */
/*                                                                           */
/* Service de génération d'un maillage à-la 'sod'.                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_SODMESHGENERATOR_H
#define ARCANE_STD_SODMESHGENERATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/std/IMeshGenerator.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Génèrateur de maillage pour un tube à choc.
 *
 * Le maillage est un maillage cartésien au format non-structuré
 *
 * Le tube dispose de deux zones ZG et ZD suivant l'axe des x: la première
 * moitié des mailles est pour ZG, la suivante pour ZD.
 *
 * En parallèle, on augmente les couches suivants l'axe z. Chaque sous-domaine
 * possède \a nb_cell_z couches suivant z et partagent une couche avec le
 * sous-domaine précédent et une couche avec le suivant. De cette manière,
 * chaque sous-domaine calcule la même chose et le nombre d'itérations
 * du cas ne change pas quelle que soit le nombre de processeurs.
 * Pour les conditions aux limites, six surfaces sont créées: XMIN, XMAX pour
 * les faces suivant X, YMIN et YMAX pour les faces suivant Y et ZMIN et
 * ZMAX pour celles suivant Z.
 */
class SodMeshGenerator
: public TraceAccessor
, public IMeshGenerator
{
 public:

  class Impl;

 public:

  SodMeshGenerator(IPrimaryMesh* tm, bool use_zxy = false);
  ~SodMeshGenerator();

 public:

  IntegerConstArrayView communicatingSubDomains() const override;
  bool readOptions(XmlNode node) override;
  bool generateMesh() override;

 private:

  IPrimaryMesh* m_mesh;
  bool m_zyx_generate;
  std::unique_ptr<Impl> m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
