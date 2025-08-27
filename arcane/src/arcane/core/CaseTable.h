// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseTable.h                                                 (C) 2000-2023 */
/*                                                                           */
/* Classe gérant une table de marche.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CASETABLE_H
#define ARCANE_CASETABLE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/datatype/SmallVariant.h"

#include "arcane/core/CaseFunction.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CaseTableParams;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Fonction du jeu de donnée.
 */
class ARCANE_CORE_EXPORT CaseTable
: public CaseFunction
{
 public:

  /*!
   * \brief Type des erreurs retournées par la classe.
   */
  enum eError
  {
    ErrNo,
    //! Indique qu'un indice d'un élément n'est pas valide
    ErrBadRange,
    //! Indique que la conversion du paramètre vers le type souhaité est impossible
    ErrCanNotConvertParamToRightType,
    //! Indique que la conversion de la valeur vers le type souhaité est impossible
    ErrCanNotConvertValueToRightType,
    //! Indique que le paramètre n'est pas supérieur au précédent
    ErrNotGreaterThanPrevious,
    //! Indique que le paramètre n'est pas inférieur au suivant
    ErrNotLesserThanNext
  };

  /*! \brief Type de la courbe de la table.
   */
  enum eCurveType
  {
    CurveUnknown = 0, //!< Type de courbe inconnu
    CurveConstant = 1, //!< Courbe constante par morceau
    CurveLinear = 2  //!< Courbe linéaire par morceau
  };

 public:

  /*! \brief Construit une table de marche du jeu de données.
   * \param curve_type type de la courbe de la table de marche
   */
  CaseTable(const CaseFunctionBuildInfo& info,eCurveType curve_type);
  virtual ~CaseTable();

 public:

  //! Nombre d'éléments de la fonction
  virtual Integer nbElement() const;

  //! \id ième valeur dans la chaîne \a str
  virtual void valueToString(Integer id,String& str) const;

  //! \id ième paramètre dans la chaîne \a str
  virtual void paramToString(Integer id,String& param) const;

   /*!
   * \brief Modifie le paramètre de l'élément \a id.
   *
   * Utilise \a value comme nouvelle valeur pour le paramètre.
   * \a value doit pourvoir être converti en le type du paramètre.
   *
   * \return la valeur de l'erreur, ErrNo sinon.
   */
  virtual eError setParam(Integer id,const String& value);

   /*!
   * \brief Modifie la valeur de l'élément \a id.
   *
   * Utilise \a value comme nouvelle valeur.
   * \a value doit pourvoir être converti en le type de la valeur.
   *
   * \return la valeur de l'erreur, ErrNo sinon.
   */
  virtual eError setValue(Integer id,const String& value);

  /*! \brief Ajoute un élément à la table.
   *
   * Ajoute l'élément (param,value) à la table.
   *
   * \return la valeur de l'erreur, ErrNo sinon.
   */
  virtual eError appendElement(const String& param,const String& value);
  
   /*!
   * \brief Insère un couple (paramètre,valeur) dans la fonction.
   *
   * Insère à la position \a id un couple (paramètre,valeur) identique
   * à celui qui s'y trouvait. Les paramètres suivants sont décalées d'un cran.
   * Il est ensuite possible de modifier ce couple par les
   * méthodes setParam() ou setValue().
   *
   * Si \a id est supérieur au nombre d'éléments de la fonction, un élément
   * est ajouté à la fin avec la même valeur que le dernier élément de
   * la fonction.
   */
  virtual void insertElement(Integer id);
  
  /*!
   * \brief Supprime un couple (paramètre,valeur) dans la fonction.
   *
   * Si \a id est supérieur au nombre d'éléments de la fonction, aucune
   * opération n'est effectuée.
   */
  virtual void removeElement(Integer id);

  /*! @name Type de la courbe */
  //@{
  //! Retourne le type de la courbe de la fonction
  virtual eCurveType curveType() const { return m_curve_type; }
  //@}

  virtual void setParamType(eParamType type);

  virtual bool checkIfValid() const;

  virtual void value(Real param,Real& v) const;
  virtual void value(Real param,Integer& v) const;
  virtual void value(Real param,bool& v) const;
  virtual void value(Real param,String& v) const;
  virtual void value(Real param,Real3& v) const;
  virtual void value(Integer param,Real& v) const;
  virtual void value(Integer param,Integer& v) const;
  virtual void value(Integer param,bool& v) const;
  virtual void value(Integer param,String& v) const;
  virtual void value(Integer param,Real3& v) const;

 public:

 private:

  CaseTableParams* m_param_list;
  UniqueArray<SmallVariant> m_value_list; //!< Liste des valeurs.
  eCurveType m_curve_type; //!< Type de la courbe
  bool m_use_fast_search = true;

 private:

  template<typename U,typename V> void _findValue(U param,V& value) const;
  template<typename U,typename V> void _findValueAndApplyTransform(U param,V& value) const;

  bool _isValidIndex(Integer index) const;
  eError _setValue(Integer index,const String& value_str);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
