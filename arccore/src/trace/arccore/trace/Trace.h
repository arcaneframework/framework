// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Trace.h                                                     (C) 2000-2025 */
/*                                                                           */
/* Traces.                                                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_TRACE_TRACE_H
#define ARCCORE_TRACE_TRACE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/trace/TraceGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Namespace contenant les types liés aux traces.
 */
namespace Trace
{
  //! Niveau de verbosité par défaut.
  static const int DEFAULT_VERBOSITY_LEVEL = 3;

  //! Niveau de verbosité non spécifié.
  static const int UNSPECIFIED_VERBOSITY_LEVEL = -2;

  //! Flot sur lequel on envoie les messages.
  enum eMessageType
  {
    Normal=0,
    Info=1,
    Warning=2,
    Error=3,
    Log=4,
    Fatal=5,
    ParallelFatal=6,
    Debug=7,
    Null=8
  };

  //! Niveau de debug des traces
  enum eDebugLevel
  {
    None = 0, //! Pas de d'informations de débug
    Lowest = 1, //!< Niveau le plus faible
    Low = 2, //!< Niveau faible
    Medium = 3, //!< Niveau moyen (défaut)
    High = 4, //!< Niveau élevé
    Highest = 5 //!< Niveau le plus élevé
  };

  //! Paramêtres d'affichage.
  enum ePrintFlags
  {
    PF_Default = 0,
    PF_NoClassName = 1 << 1, //!< Affichage ou non de la classe de message.
    PF_ElapsedTime = 1 << 2, //!< Affichage du temps écoulé
  };

  /*!
   * \brief Formattage du flot en longueur.
   *
   * \sa TraceMessage, ITraceMng
   */
  class ARCCORE_TRACE_EXPORT Width
  {
   public:
    //! Formatte le flot sur une longueur de \a v caractères.
    Width(Integer v) : m_width(v) {}
   public:
    Integer m_width; //!< Longueur du formattage.
  };
  /*!
   * \brief Formattage des réels avec une précision donnée.
   *
   * \sa TraceMessage, ITraceMng
   */
  class ARCCORE_TRACE_EXPORT Precision
  {
   public:
    //! Imprime la valeur \a value avec \a n chiffres significatifs.
    Precision(Integer n,Real value)
    : m_precision(n), m_value(value), m_scientific(false) {}
    /*! Imprime la valeur \a value avec \n n chiffres significatifs en mode scienfique si
     * \a scientific est vrai
     */
    Precision(Integer n,Real value,bool scientific)
    : m_precision(n), m_value(value), m_scientific(scientific) {}
   public:
    Integer m_precision; //!< Nombre de chiffres significatifs.
    Real m_value; //!< Valeur à sortir
    bool m_scientific; //! True si sortie en mode scientifique
  };

  /*!
   * \brief Positionne une couleur pour le message
   *
   * \sa TraceMessage, ITraceMng
   */
  class ARCCORE_TRACE_EXPORT Color
  {
   public:
    // Liste des couleurs.
    // NOTE: tout changement ici doit être reporté dans TraceMng.cc
    enum { Black = 0,
           DarkRed, DarkGreen, DarkYellow, DarkBlue, DarkMagenta, DarkCyan, DarkGrey,
           Red, Green, Yellow, Blue, Magenta, Cyan, Grey };

    static const int LAST_COLOR = Grey;
   public:
    Color(int color) : m_color(color){}
   public:
    static Color darkRed() { return Color(DarkRed); }
    static Color darkGreen() { return Color(DarkGreen); }
    static Color darkYellow() { return Color(DarkYellow); }
    static Color darkBlue() { return Color(DarkBlue); }
    static Color darkMagenta() { return Color(DarkMagenta); }
    static Color darkCyan() { return Color(DarkCyan); }
    static Color darkGrey() { return Color(DarkGrey); }

    static Color red() { return Color(Red); }
    static Color green() { return Color(Green); }
    static Color yellow() { return Color(Yellow); }
    static Color blue() { return Color(Blue); }
    static Color magenta() { return Color(Magenta); }
    static Color cyan() { return Color(Cyan); }
    static Color grey() { return Color(Grey); }
   public:
    int m_color;
  };


  /*!
   * \brief Positionne une classe de message.
   *
   * Positionne pour la classe de message \a name au moment de la construction
   * de l'instance et remet l'ancienne au moment de sa destruction.
   *
   * \sa TraceMessage, ITraceMng
   */
  class ARCCORE_TRACE_EXPORT Setter
  {
   public:
    //! Positionne la classe \a name pour le gestionnaire \a msg
    Setter(ITraceMng* msg,const String& name);
    //! Libère l'instance et remet l'ancienne classe de message dans \a m_msg
    ~Setter();
   public:
   private:
    ITraceMng* m_msg; //!< Gestionnaire de message.
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

