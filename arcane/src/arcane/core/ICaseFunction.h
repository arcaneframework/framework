// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICaseFunction.h                                             (C) 2000-2023 */
/*                                                                           */
/* Interface d'une fonction du jeu de données.                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ICASEFUNCTION_H
#define ARCANE_CORE_ICASEFUNCTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 *
 * \brief Interface d'une fonction du jeu de données.
 *
 * \ingroup CaseOption
 *
 * Une fonction du jeu de données est une fonction mathématique f(x)->y avec
 * \c x le \e paramètre et \c y la \e valeur.
 *
 * Dans la version actuelle, une fonction est décrite par morceaux par
 * un ensemble de couples (x,y).
 *
 * Les méthodes qui permettent d'éditer cette table de marche sont utilisées
 * principalement par l'éditeur du jeu de données. Dans tous les cas,
 * elles ne doivent pas être appelées une fois que le jeu de données complet
 * a été lu (ICaseMng::readCaseOptions).
 */
class ARCANE_CORE_EXPORT ICaseFunction
{
  ARCCORE_DECLARE_REFERENCE_COUNTED_INCLASS_METHODS();

 public:

  /*!
   * \brief Type d'un paramètre d'une fonction.
   */
  enum eParamType
  {
    ParamUnknown =0, //!< Type de paramètre inconnu
    ParamReal = 1, //!< Paramètre de type Real
    ParamInteger = 2 //!< Paramètre de type Integer
  };
  /*!
   * \brief Type d'une valeur d'une fonction
   */
  enum eValueType
  {
    ValueUnknown = 0, //!< Type de la valeur inconnu
    ValueReal = 1, //!< Valeur de type réelle
    ValueInteger = 2, //!< Valeur de type entière
    ValueBool = 3, //!< Valeur de type entière
    ValueString = 4,  //!< Valeur de type chaîne de caractères
    ValueReal3 = 5  //!< Valeur de type 'Real3'
  };

 public:
	
  // NOTE: Laisse temporairement ce destructeur publique tant
  // qu'on appelle explicitement ce destructeur mais avec le compteur
  // de référence cela ne sera normalement plus le cas.
  virtual ~ICaseFunction() = default; //!< Libère les ressources

 public:

  /*! @name Nom de la fonction */
  //@{
  //! nom de la fonction
  virtual String name() const =0;

  //! Positionne le nom de la fonction en \a new_name
  virtual void setName(const String& new_name) =0;
  //@}

  /*! @name Type du paramètre */
  //@{
  //! Type du paramètre de la fonction
  virtual eParamType paramType() const =0;

  //! Positionne le type de paramètre de la fonction
  virtual void setParamType(eParamType type) =0;
  //@}

  /*! @name Type de la valeur */
  //@{
  //! Type des valeurs de la fonction
  virtual eValueType valueType() const =0;

  //! Positionne le type des valeurs de la fonction
  virtual void setValueType(eValueType type) =0;
  //@}

  /*!
   * \brief Affecte une fonction de transformation de la valeur.
   * Pour l'instant, il s'agit juste d'un coefficient multiplicatif.
   * La chaîne \a str doit pouvoir être convertie en le type de la valeur.
   */
  virtual void setTransformValueFunction(const String& str) =0;

  //! Retourne la fonction de transformation de la valeur.
  virtual String transformValueFunction() const =0;

  /*!
   * \brief Affecte une fonction de transformation du paramètre.
   * Pour l'instant, il s'agit juste d'un coefficient multiplicatif.
   * Il n'est appliqué que pour les paramètre réels.
   * La chaîne \a str doit pouvoir être convertie en un réel.
   */
  virtual void setTransformParamFunction(const String& str) =0;

  //! Fonction de transformation du paramètre
  virtual String transformParamFunction() const =0;

  /*!
   * \brief Vérifie la validité de la fonction.
   * \retval true si la fonction est valide,
   * \retval false sinon.
   */
  virtual bool checkIfValid() const =0;

  /*!
   * \brief Positionne la Valeur du coefficient multiplicateur du deltat.
   *
   * Ce coefficient, 0.0 par défaut est utilisé pour les fonctions
   * qui prennent en paramètre le temps physique. Dans ce cas,
   * la fonction utilise comme paramètre le temps courant global
   * auquel est ajouté le pas de temps courant global multiplié
   * par ce coefficient.
   */
  virtual void setDeltatCoef(Real v) =0;

  //! Valeur du coefficient multiplicateur du deltat
  virtual Real deltatCoef() const =0;

 public:

  //! Valeur \a v de l'option pour le paramètre \a param.
  virtual void value(Real param,Real& v) const =0;

  //! Valeur \a v de l'option pour le paramètre \a param.
  virtual void value(Real param,Integer& v) const =0;

  //! Valeur \a v de l'option pour le paramètre \a param.
  virtual void value(Real param,bool& v) const =0;

  //! Valeur \a v de l'option pour le paramètre \a param.
  virtual void value(Real param,String& v) const =0;

  //! Valeur \a v de l'option pour le paramètre \a param.
  virtual void value(Real param,Real3& v) const =0;

  //! Valeur \a v de l'option pour le paramètre \a param.
  virtual void value(Integer param,Real& v) const =0;

  //! Valeur \a v de l'option pour le paramètre \a param.
  virtual void value(Integer param,Integer& v) const =0;

  // Valeur \a v de l'option pour le paramètre \a param.
  virtual void value(Integer param,bool& v) const =0;

  //! Valeur \a v de l'option pour le paramètre \a param.
  virtual void value(Integer param,String& v) const =0;

  //! Valeur \a v de l'option pour le paramètre \a param.
  virtual void value(Integer param,Real3& v) const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
