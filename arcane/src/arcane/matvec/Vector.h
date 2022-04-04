// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Vector.h                                                    (C) 2000-2007 */
/*                                                                           */
/* Vecteur d'algèbre linéraire.                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATVEC_VECTOR_H
#define ARCANE_MATVEC_VECTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/Array.h"
#include "arcane/utils/Numeric.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
namespace MatVec
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VectorImpl;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vecteur d'algèbre linéraire.
 */
class ARCANE_CORE_EXPORT Vector
{
 public:
  //! Créé un vecteur vide
  Vector();
  /*!
   * \brief Créé pour stocker \a size éléments.
   *
   * Le vecteur n'est pas initialisé et ses valeurs sont quelconques.
   */  
  explicit Vector(Integer size);
  /*!
   * \brief Créé pour stocker \a size éléments.
   *
   * Le vecteur est initialisé avec les valeurs de \a init_value.
   */  
  Vector(Integer size,Real init_value);
  /*!
   * \brief Créé un vecteur avec les éléments de \a v.
   */
  Vector(RealUniqueArray v);
  /*!
   * \brief Construit un vecteur qui référence \a rhs.
   */
  Vector(const Vector& rhs);
  //! Change la référence du vecteur
  const Vector& operator=(const Vector& rhs);
  //! Supprime la référence
  ~Vector();
 public:
  //! Nombre d'éléments du vecteur
  Integer size() const;
  /*!
   * \brief Valeurs du vecteur.
   * \warning la vue retournée est invalidée dès que le vecteur
   * est redimensionné.
   */
  RealArrayView values();
  /*!
   * \brief Valeurs du vecteur
   * \warning la vue retournée est invalidée dès que le vecteur
   * est redimensionné.
   */
  RealConstArrayView values() const;
  //! Imprime les valeurs du vecteur
  void dump(std::ostream& o) const;
  //! Clone ce vecteur
  Vector clone();
  /*!
   * \brief Copie les éléments de \a rhs dans ce vecteur.
   *
   * Le vecteur peut éventuellement être redimensionné.
   */
  void copy(const Vector& rhs);
  /*!
   * \brief Change le nombre d'éléments du vecteur.
   *
   * Si le nombre d'éléments augmente, les nouveaux éléments ne sont
   * pas initialisé.
   */
  void resize(Integer new_size);
  /*!
   * \brief Change le nombre d'éléments du vecteur.
   *
   * Si le nombre d'éléments augmente, les nouveaux éléments sont
   * initialisé avec la valeur \a init_value.
   */
  void resize(Integer new_size,Real init_value);

  Real normInf();

  //! Initialise un vecteur en utilisant un fichier au format Hypre
  static Vector readHypre(const String& file_name);
 private:
  //! Représentation interne du groupe.
  VectorImpl* m_impl;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
}
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
