// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* InversiveCongruential.h                                     (C) 2000-2025 */
/*                                                                           */
/* Ce fichier définit le patron de classe InversiveCongruential ainsi qu'une */
/* classe associée Hellekalek1995.  Il est une version adaptée du fichier    */
/* InversiveCongruential.hpp provenant de la bibliothèque BOOST              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_RANDOM_INVERSIVECONGRUENTIAL_H
#define ARCANE_CORE_RANDOM_INVERSIVECONGRUENTIAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/random/RandomGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::random
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*! Patron de classe \c InversiveCongruential. Il permet de définir des classes  
 * de générateurs de type Inversive Congruential en fonction des paramètres \c a,
 * \c c et \c m. Les nombres pseudo-aléatoires générés sont de type \c IntType.
 *
 * La génération d'une séquence de  nombres pseudo-aléatoires s'effectue :
 *
 * - soit par l'appel successif de l'opérateur \c (). Dans ce cas, la graine peut  
 *   être initialisée par l'appel au constructeur  ou les différentes méthodes  
 *   \c seed . L'état du générateur est gérée en interne par l'intermédiaire du 
 *   membre private \c _x. Sa valeur est accessible via la méthode \c getState().
 *
 * - soit par l'appel de la méthode \c apply(x). L'état du générateur \c x 
 *   est géré à l'extérieur de la classe. Les méthodes \c seed et \c getState() 
 *   n'ont pas de sens dans cette utilisation.
*/
template<typename IntType, IntType a, IntType c, IntType m, IntType val>
class InversiveCongruential
{
 public:
  typedef IntType result_type;
  static const bool has_fixed_range = true;
  static const result_type min_value = ( c == 0 ? 1 : 0 );
  static const result_type max_value = m-1;
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief Retourne la valeur minimum possible d'une séquence. 
   *
   * \author Patrick Rathouit (origine bibliotheque BOOST)
   * \date 28/07/2006
   */
  result_type min() const { return c == 0 ? 1 : 0; }
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief Retourne la valeur maximum possible d'une séquence. 
   *
   * \author Patrick Rathouit (origine bibliotheque BOOST)
   * \date 28/07/2006
   */
  result_type max() const { return m-1; }
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief Constructeur avec initialisation de la graine à partir de la valeur
   *         \c x0.
   *
   * \author Patrick Rathouit (origine bibliotheque BOOST)
   * \date 28/07/2006
   */  
  explicit InversiveCongruential(IntType x0 = 1)
    : _x(x0)
  { 
  }
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief Initialisation de la graine du générateur à partir de la valeur \c x0.
   *
   * \author Patrick Rathouit (origine bibliotheque BOOST)
   * \date 28/07/2006
   */
  void seed(IntType x0) { _x = x0; }
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief Méthode qui retourne l'état générateur.
   *
   * \author Patrick Rathouit (origine bibliotheque BOOST)
   * \date   28/07/2006
   */
  IntType getState() const { return _x; }
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief Surdéfinition de l'opérateur () qui retourne la valeur pseudo 
   *         aléatoire du générateur. L'état du générateur est modifié. 
   *
   * \author Patrick Rathouit (origine bibliotheque BOOST)
   * \date 28/07/2006
   */
  IntType operator()()
  {
    _x = apply(_x);
    return _x;
  }
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*! \brief Retourne la valeur pseudo aléatoire à partir de l'état \c x. Le 
 *         membre privée \c _x du générateur n'est pas utilisé et n'est pas  
 *         modifié. 
 *
 * \author Patrick Rathouit (origine bibliotheque BOOST)
 * \date  28/07/2006
 */
  static IntType apply(IntType x)
  {
    typedef utils::const_mod<IntType, m> do_mod;
    return x = do_mod::mult_add(a,do_mod::invert(x), c);
  }
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief Fonction de validation (je ne sais pas trop a quoi elle sert!)
   *
   * \author Patrick Rathouit (origine bibliotheque BOOST)
   * \date 28/07/2006
   */
  bool validation(IntType x) const { return val == x; }
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*! \brief Surdéfinition de l'opérateur ==
   *
   * \author Patrick Rathouit (origine bibliotheque BOOST)
   * \date 28/07/2006
   */
  bool operator==(const InversiveCongruential& rhs) const
    { return _x == rhs._x; }

 private:

  IntType _x;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

typedef InversiveCongruential<Int32, 9102, 2147483647-36884165,
  2147483647, 0> Hellekalek1995;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

