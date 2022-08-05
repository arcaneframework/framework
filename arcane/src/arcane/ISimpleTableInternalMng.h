// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ISimpleTableInternalMng.h                                   (C) 2000-2022 */
/*                                                                           */
/* Interface représentant un gestionnaire de SimpleTableInternal. Fichier    */
/* contenant également la structure SimpleTableInternal.                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_ISIMPLETABLEINTERNALMNG_H
#define ARCANE_ISIMPLETABLEINTERNALMNG_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ItemTypes.h"
#include "arcane/ISubDomain.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/Array2.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO : Voir pour ajouter ISharedReference.

/**
 * @brief Structure représentant un tableau simple.
 * 
 * Un tableau simple ressemble à ça :
 * 
 * NomTableau | C1 | C2 | C3
 *    L1      |Val1|Val2|Val3
 *    L2      |Val4|Val5|Val6
 * 
 * Un nom de tableau, une liste de noms de lignes,
 * une liste de noms de colonnes et une liste 2D
 * de valeur (Real pour l'instant).
 * 
 */
struct ARCANE_CORE_EXPORT SimpleTableInternal
{
  SimpleTableInternal(ISubDomain* sub_domain)
  : m_sub_domain(sub_domain)
  {
  }
  ~SimpleTableInternal() = default;

  void clear()
  {
    m_values_csv.clear();
    m_name_rows.clear();
    m_name_columns.clear();
    m_name_tab = "";
    m_size_rows.clear();
    m_size_columns.clear();
    m_last_row = -1;
    m_last_column = -1;
  }

  UniqueArray2<Real> m_values_csv;

  UniqueArray<String> m_name_rows;
  UniqueArray<String> m_name_columns;

  String m_name_tab;

    // Tailles des lignes/colonnes
  // (et pas le nombre d'éléments, on compte les "trous" entre les éléments ici,
  // mais sans le trou de fin).
  // Ex. : {{"1", "2", "0", "3", "0", "0"},
  //        {"4", "5", "6", "0", "7", "8"},
  //        {"0", "0", "0", "0", "0", "0"}}

  //       m_size_rows[0] = 4
  //       m_size_rows[1] = 6
  //       m_size_rows[2] = 0
  //       m_size_rows.size() = 3

  //       m_size_columns[3] = 1
  //       m_size_columns[0; 1; 2; 4; 5] = 2
  //       m_size_columns.size() = 6
  UniqueArray<Integer> m_size_rows;
  UniqueArray<Integer> m_size_columns;

  // Dernier élement ajouté.
  Integer m_last_row;
  Integer m_last_column;

  ISubDomain* m_sub_domain;
};

/**
 * @brief Interface de classe représentant un gestionnaire
 * de SimpleTableInternal (aka STI). 
 * 
 * Ce gestionnaire permet de faire plusieurs types d'opérations
 * sur le STI : ajout de lignes, de colonnes, de valeurs, &c.
 * 
 * Il y a deux modes d'exploitations (qui peuvent être mélangés) : 
 * - en utilisant les noms ou positions des lignes/colonnes,
 * - en utilisant un pointeur de position dans le tableau.
 * 
 * Le premier mode est le plus simple à utiliser et est suffisant
 * pour la plupart des utilisateurs. On donne un nom (ou une position)
 * de ligne ou de colonne et une valeur, et cette valeur est placée
 * à la suite des autres valeurs sur la ligne ou sur la colonne.
 * 
 * Le second mode est plus avancé et sert surtout à remplacer des
 * élements déjà présent ou à optimiser les performances (s'il y a 
 * 40 lignes, 40 valeurs à ajouter à la suite et qu'on utilise les 
 * noms des colonnes 40 fois, cela fait 40 recherches de String dans un 
 * StringUniqueArray, ce qui n'est pas top niveau optimisation).
 * Un pointeur représentant le dernier élement ajouté est présent dans
 * STI. On peut modifier les élements autour de ce pointeur (haut, bas
 * gauche, droite) avec les méthodes présentes.
 * Ce pointeur peut être placé n'importe où grâce au méthodes elem().
 * Ce pointeur n'est pas lu par les méthodes du premier mode mais est
 * mis à jour par ces dernières.
 */
class ARCANE_CORE_EXPORT ISimpleTableInternalMng
{
public:
  virtual ~ISimpleTableInternalMng() = default;

public:

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant d'effacer le contenu
   * du SimpleTableInternal.
   */
  virtual void clearInternal() = 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant d'ajouter une ligne.
   * 
   * @param name_row Le nom de la ligne.
   * @return Integer La position de la ligne dans le tableau.
   */
  virtual Integer addRow(const String& name_row) = 0;
  /**
   * @brief Méthode permettant d'ajouter une ligne.
   * 
   * Si le nombre d'élements dans 'elems' est plus grand que le
   * nombre de colonnes, l'ajout s'effectue quand même (mais les
   * éléments en trop ne seront pas ajoutés).
   * 
   * @param name_row Le nom de la ligne.
   * @param elems Les éléments à insérer sur la ligne.
   * @return Integer La position de la ligne dans le tableau.
   */
  virtual Integer addRow(const String& name_row, ConstArrayView<Real> elems) = 0;
  /**
   * @brief Méthode permettant d'ajouter plusieurs lignes.
   * 
   * @param name_rows Les noms des lignes.
   * @return true Si toutes les lignes ont été créées.
   * @return false Si toutes les lignes n'ont pas été créées.
   */
  virtual bool addRows(StringConstArrayView name_rows) = 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant d'ajouter une colonne.
   * 
   * @param name_column Le nom de la colonne.
   * @return Integer La position de la colonne dans le tableau.
   */
  virtual Integer addColumn(const String& name_column) = 0;
  /**
   * @brief Méthode permettant d'ajouter une colonne.
   * 
   * Si le nombre d'élements dans 'elems' est plus grand que le
   * nombre de lignes, l'ajout s'effectue quand même (mais les
   * éléments en trop ne seront pas ajoutés).
   * 
   * @param name_column Le nom de la colonne.
   * @param elems Les éléments à ajouter sur la colonne.
   * @return Integer La position de la colonne dans le tableau.
   */
  virtual Integer addColumn(const String& name_column, ConstArrayView<Real> elems) = 0;
  /**
   * @brief Méthode permettant d'ajouter plusieurs colonnes.
   * 
   * @param name_rows Les noms des colonnes.
   * @return true Si toutes les colonnes ont été créées.
   * @return false Si toutes les colonnes n'ont pas été créées.
   */
  virtual bool addColumns(StringConstArrayView name_columns) = 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant d'ajouter un élément à une ligne.
   * 
   * @param pos La position de la ligne.
   * @param elem L'élément à ajouter.
   * @return true Si l'élément a pu être ajouté.
   * @return false Si l'élément n'a pas pu être ajouté.
   */
  virtual bool addElemRow(Integer pos, Real elem) = 0;
  /**
   * @brief Méthode permettant l'ajouter un élément sur une ligne.
   * 
   * @param name_row Le nom de la ligne.
   * @param elem L'élément à ajouter.
   * @param create_if_not_exist Pour savoir si l'on doit créer la 
   *                            ligne si elle n'existe pas encore.
   * @return true Si l'élément a pu être ajouté.
   * @return false Si l'élément n'a pas pu être ajouté.
   */
  virtual bool addElemRow(const String& name_row, Real elem, bool create_if_not_exist = true) = 0;
  /**
   * @brief Méthode permettant d'ajouter un élément sur la ligne 
   * dernièrement manipulée.
   * 
   * Cette méthode diffère de 'editElemRight()' car ici, on ajoute 
   * un élément à la fin de la ligne, pas forcement après le
   * dernier élement ajouté.
   * 
   * @param elem L'élément à ajouter.
   * @return true Si l'élément a été ajouté.
   * @return false Si l'élément n'a pas pu être ajouté.
   */
  virtual bool addElemSameRow(Real elem) = 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant d'ajouter plusieurs éléments sur une ligne.
   * 
   * Si le nombre d'élements dans 'elems' est plus grand que le
   * nombre de colonnes disponibles, l'ajout s'effectue quand même (mais les
   * éléments en trop ne seront pas ajoutés) et on aura un return false.
   * 
   * @param pos La position de la ligne.
   * @param elems Le tableau d'élement à ajouter.
   * @return true Si tous les éléments ont été ajoutés.
   * @return false Si [0;len(elems)[ éléments ont été ajoutés.
   */
  virtual bool addElemsRow(Integer pos, ConstArrayView<Real> elems) = 0;
  /**
   * @brief Méthode permettant d'ajouter plusieurs éléments sur une ligne.
   * 
   * Si le nombre d'élements dans 'elems' est plus grand que le
   * nombre de colonnes disponibles, l'ajout s'effectue quand même (mais les
   * éléments en trop ne seront pas ajoutés) et on aura un return false.
   * 
   * @param name_row Le nom de la ligne.
   * @param elems Le tableau d'élement à ajouter.
   * @param create_if_not_exist Pour savoir si l'on doit créer la ligne
   *                            si elle n'existe pas encore.
   * @return true Si tous les éléments ont été ajoutés.
   * @return false Si [0;len(elems)[ éléments ont été ajoutés.
   */
  virtual bool addElemsRow(const String& name_row, ConstArrayView<Real> elems, bool create_if_not_exist = true) = 0;
  /**
   * @brief Méthode permettant d'ajouter plusieurs éléments sur la 
   * ligne dernièrement manipulée.
   * 
   * Si le nombre d'élements dans 'elems' est plus grand que le
   * nombre de colonnes disponibles, l'ajout s'effectue quand même (mais les
   * éléments en trop ne seront pas ajoutés) et on aura un return false.
   * 
   * Mis à part le fait qu'ici, on manipule un tableau, cette méthode diffère
   * de 'editElemRight()' car ici, on ajoute des éléments à la fin de la ligne,
   * pas forcement après le dernier élement ajouté.
   * 
   * @param elems Le tableau d'élement à ajouter.
   * @return true Si tous les éléments ont été ajoutés.
   * @return false Si [0;len(elems)[ éléments ont été ajoutés.
   */
  virtual bool addElemsSameRow(ConstArrayView<Real> elems) = 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant d'ajouter un élément à une colonne.
   * 
   * @param pos La position de la colonne.
   * @param elem L'élément à ajouter.
   * @return true Si l'élément a pu être ajouté.
   * @return false Si l'élément n'a pas pu être ajouté.
   */
  virtual bool addElemColumn(Integer pos, Real elem) = 0;
  /**
   * @brief Méthode permettant l'ajouter un élément sur une colonne.
   * 
   * @param name_column Le nom de la colonne.
   * @param elem L'élément à ajouter.
   * @param create_if_not_exist Pour savoir si l'on doit créer la colonne
   *                            si elle n'existe pas encore.
   * @return true Si l'élément a pu être ajouté.
   * @return false Si l'élément n'a pas pu être ajouté.
   */
  virtual bool addElemColumn(const String& name_column, Real elem, bool create_if_not_exist = true) = 0;
  /**
   * @brief Méthode permettant d'ajouter un élément sur la colonne
   * dernièrement manipulée.
   * 
   * Cette méthode diffère de 'editElemDown()' car ici, on ajoute un élément
   * à la fin de la colonne, pas forcement après le dernier élement ajouté.
   * 
   * @param elem L'élément à ajouter.
   * @return true Si l'élément a été ajouté.
   * @return false Si l'élément n'a pas pu être ajouté.
   */
  virtual bool addElemSameColumn(Real elem) = 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant d'ajouter plusieurs éléments sur une colonne.
   * 
   * Si le nombre d'élements dans 'elems' est plus grand que le
   * nombre de lignes disponibles, l'ajout s'effectue quand même (mais les
   * éléments en trop ne seront pas ajoutés) et on aura un return false.
   * 
   * @param pos La position de la colonne.
   * @param elems Le tableau d'élement à ajouter.
   * @return true Si tous les éléments ont été ajoutés.
   * @return false Si [0;len(elems)[ éléments ont été ajoutés.
   */
  virtual bool addElemsColumn(Integer pos, ConstArrayView<Real> elems) = 0;
  /**
   * @brief Méthode permettant d'ajouter plusieurs éléments sur une colonne.
   * 
   * Si le nombre d'élements dans 'elems' est plus grand que le
   * nombre de lignes disponibles, l'ajout s'effectue quand même (mais les
   * éléments en trop ne seront pas ajoutés) et on aura un return false.
   * 
   * @param name_column Le nom de la colonne.
   * @param elems Le tableau d'élement à ajouter.
   * @param create_if_not_exist Pour savoir si l'on doit créer la colonne si
   *                            elle n'existe pas encore.
   * @return true Si tous les éléments ont été ajoutés.
   * @return false Si [0;len(elems)[ éléments ont été ajoutés.
   */
  virtual bool addElemsColumn(const String& name_column, ConstArrayView<Real> elems, bool create_if_not_exist = true) = 0;
  /**
   * @brief Méthode permettant d'ajouter plusieurs éléments sur la
   * colonne dernièrement manipulée.
   * 
   * Si le nombre d'élements dans 'elems' est plus grand que le
   * nombre de lignes disponibles, l'ajout s'effectue quand même (mais les
   * éléments en trop ne seront pas ajoutés) et on aura un return false.
   * 
   * Mis à part le fait qu'ici, on manipule un tableau, cette méthode diffère
   * de 'editElemDown()' car ici, on ajoute des éléments à la fin de la colonne,
   * pas forcement après le dernier élement ajouté.
   * 
   * @param elems Le tableau d'élement à ajouter.
   * @return true Si tous les éléments ont été ajoutés.
   * @return false Si [0;len(elems)[ éléments ont été ajoutés.
   */
  virtual bool addElemsSameColumn(ConstArrayView<Real> elems) = 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant d'éditer un élément au-dessus du dernier
   * élement dernièrement manipulé (ligne du dessus/même colonne).
   * 
   * L'élement que l'on modifie devient donc le dernier élement modifié
   * à la fin de cette méthode (si update_last_pos = true).
   * 
   * @param elem L'élement à modifier.
   * @param update_last_pos Doit-on déplacer le curseur "dernier élement modifié" ?
   * @return true Si l'élément a été modifié.
   * @return false Si l'élément n'a pas pu être modifié.
   */
  virtual bool editElemUp(Real elem, bool update_last_pos = true) = 0;
  /**
   * @brief Méthode permettant d'éditer un élément en-dessous du dernier 
   * élement dernièrement manipulé (ligne du dessous/même colonne).
   * 
   * L'élement que l'on modifie devient donc le dernier élement modifié 
   * à la fin de cette méthode (si update_last_pos = true).
   * 
   * Cette méthode diffère de 'addElemSameColumn()' car ici, on ajoute 
   * (ou modifie) un élement sous le dernier élement manipulé, qui n'est
   * pas forcement à la fin de la colonne.
   * 
   * @param elem L'élement à modifier.
   * @param update_last_pos Doit-on déplacer le curseur "dernier élement modifié" ?
   * @return true Si l'élément a été modifié.
   * @return false Si l'élément n'a pas pu être modifié.
   */
  virtual bool editElemDown(Real elem, bool update_last_pos = true) = 0;
  /**
   * @brief Méthode permettant d'éditer un élément à gauche du dernier
   * élement dernièrement manipulé (même ligne/colonne à gauche).
   * 
   * L'élement que l'on modifie devient donc le dernier élement modifié
   * à la fin de cette méthode (si update_last_pos = true).
   * 
   * @param elem L'élement à modifier.
   * @param update_last_pos Doit-on déplacer le curseur "dernier élement modifié" ?
   * @return true Si l'élément a été modifié.
   * @return false Si l'élément n'a pas pu être modifié.
   */
  virtual bool editElemLeft(Real elem, bool update_last_pos = true) = 0;
  /**
   * @brief Méthode permettant d'éditer un élément à droite du dernier
   * élement dernièrement manipulé (même ligne/colonne à droite).
   * 
   * L'élement que l'on modifie devient donc le dernier élement modifié
   * à la fin de cette méthode (si update_last_pos = true).
   * 
   * Cette méthode diffère de 'addElemSameRow()' car ici, on ajoute 
   * (ou modifie) un élement à la droite du dernier élement manipulé, 
   * qui n'est pas forcement à la fin de la colonne.
   * 
   * @param elem L'élement à modifier.
   * @param update_last_pos Doit-on déplacer le curseur "dernier élement modifié" ?
   * @return true Si l'élément a été modifié.
   * @return false Si l'élément n'a pas pu être modifié.
   */
  virtual bool editElemRight(Real elem, bool update_last_pos = true) = 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant de récupérer un élément au-dessus du dernier
   * élement dernièrement manipulé (ligne du dessus/même colonne).
   * 
   * L'élement que l'on récupère devient donc le dernier élement "modifié"
   * à la fin de cette méthode (si update_last_pos = true).
   * 
   * @param update_last_pos Doit-on déplacer le curseur "dernier élement modifié" ?
   * @return Real L'élement trouvé (0 si non trouvé).
   */
  virtual Real elemUp(bool update_last_pos = false) = 0;
  /**
   * @brief Méthode permettant de récupérer un élément en-dessous du dernier
   * élement dernièrement manipulé (ligne du dessous/même colonne).
   * 
   * L'élement que l'on récupère devient donc le dernier élement "modifié"
   * à la fin de cette méthode (si update_last_pos = true).
   * 
   * @param update_last_pos Doit-on déplacer le curseur "dernier élement modifié" ?
   * @return Real L'élement trouvé (0 si non trouvé).
   */
  virtual Real elemDown(bool update_last_pos = false) = 0;
  /**
   * @brief Méthode permettant de récupérer un élément à gauche du dernier
   * élement dernièrement manipulé (même ligne/colonne à gauche).
   * 
   * L'élement que l'on récupère devient donc le dernier élement "modifié"
   * à la fin de cette méthode (si update_last_pos = true).
   * 
   * @param update_last_pos Doit-on déplacer le curseur "dernier élement modifié" ?
   * @return Real L'élement trouvé (0 si non trouvé).
   */
  virtual Real elemLeft(bool update_last_pos = false) = 0;
  /**
   * @brief Méthode permettant de récupérer un élément à droite du dernier
   * élement dernièrement manipulé (même ligne/colonne à droite).
   * 
   * L'élement que l'on récupère devient donc le dernier élement "modifié"
   * à la fin de cette méthode (si update_last_pos = true).
   * 
   * @param update_last_pos Doit-on déplacer le curseur "dernier élement modifié" ?
   * @return Real L'élement trouvé (0 si non trouvé).
   */
  virtual Real elemRight(bool update_last_pos = false) = 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant de modifier un élement dans le tableau.
   * 
   * Les positions x et y correspondent à l'emplacement du dernier
   * élement manipulé.
   * 
   * Cette méthode a de l'intéret après l'utilisation de
   * 'elemUDLR(true)' par exemple.
   * 
   * @param elem L'élement remplaçant.
   * @return true Si l'élement a bien été remplacé.
   * @return false Si l'élement n'a pas été remplacé.
   */
  virtual bool editElem(Real elem) = 0;
  /**
   * @brief Méthode permettant de modifier un élement dans le tableau.
   * 
   * @param pos_x La position de la colonne à modifier.
   * @param pos_y La position de la ligne à modifier.
   * @param elem L'élement remplaçant.
   * @return true Si l'élement a bien été remplacé.
   * @return false Si l'élement n'a pas été remplacé.
   */
  virtual bool editElem(Integer pos_x, Integer pos_y, Real elem) = 0;
  /**
   * @brief Méthode permettant de modifier un élement dans le tableau.
   * 
   * @param name_column Le nom de la colonne où se trouve l'élement.
   * @param name_row Le nom de la ligne où se trouve l'élement.
   * @param elem L'élement remplaçant.
   * @return true Si l'élement a bien été remplacé.
   * @return false Si l'élement n'a pas pu être remplacé.
   */
  virtual bool editElem(const String& name_column, const String& name_row, Real elem) = 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant d'obtenir une copie d'un élement.
   * 
   * Les positions x et y correspondent à l'emplacement du dernier élement manipulé.
   * 
   * @return Real L'élement trouvé (0 si non trouvé).
   */
  virtual Real elem() = 0;
  /**
   * @brief Méthode permettant d'obtenir une copie d'un élement.
   * 
   * @param pos_x La position de la colonne où se trouve l'élement.
   * @param pos_y La position de la ligne où se trouve l'élement.
   * @param update_last_pos Doit-on déplacer le curseur "dernier élement modifié" ?
   * @return Real L'élement trouvé (0 si non trouvé).
   */
  virtual Real elem(Integer pos_x, Integer pos_y, bool update_last_pos = false) = 0;
  /**
   * @brief Méthode permettant d'obtenir une copie d'un élement.
   * 
   * @param name_column Le nom de la colonne où se trouve l'élement.
   * @param name_row Le nom de la ligne où se trouve l'élement.
   * @param update_last_pos Doit-on déplacer le curseur "dernier élement modifié" ?
   * @return Real L'élement trouvé (0 si non trouvé).
   */
  virtual Real elem(const String& name_column, const String& name_row, bool update_last_pos = false) = 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant d'obtenir une copie d'une ligne.
   * 
   * @param pos La position de la ligne.
   * @return RealUniqueArray La copie de la ligne (tableau vide si non trouvée).
   */
  virtual RealUniqueArray row(Integer pos) = 0;
  /**
   * @brief Méthode permettant d'obtenir une copie d'une ligne.
   * 
   * @param name_row Le nom de la ligne.
   * @return RealUniqueArray La copie de la ligne (tableau vide si non trouvée).
   */
  virtual RealUniqueArray row(const String& name_row) = 0;

  /**
   * @brief Méthode permettant d'obtenir une copie d'une colonne.
   * 
   * @param pos La position de la colonne.
   * @return RealUniqueArray La copie de la colonne (tableau vide si non trouvée).
   */
  virtual RealUniqueArray column(Integer pos) = 0;
  /**
   * @brief Méthode permettant d'obtenir une copie d'une colonne.
   * 
   * @param name_column Le nom de la colonne.
   * @return RealUniqueArray La copie de la colonne (tableau vide si non trouvée).
   */
  virtual RealUniqueArray column(const String& name_column) = 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant d'obtenir la taille d'une ligne.
   * Incluant les hypothétiques 'trous' dans la ligne.
   * 
   * @param pos La position de la ligne.
   * @return Integer La taille de la ligne (0 si non trouvée).
   */
  virtual Integer sizeRow(Integer pos) = 0;
  /**
   * @brief Méthode permettant d'obtenir la taille d'une ligne.
   * Incluant les hypotétiques 'trous' dans la ligne.
   * 
   * @param pos Le nom de la ligne.
   * @return Integer La taille de la ligne (0 si non trouvée).
   */
  virtual Integer sizeRow(const String& name_row) = 0;

  /**
   * @brief Méthode permettant d'obtenir la taille d'une colonne.
   * Incluant les hypotétiques 'trous' dans la colonne.
   * 
   * @param pos La position de la colonne.
   * @return Integer La taille de la colonne (0 si non trouvée).
   */
  virtual Integer sizeColumn(Integer pos) = 0;
  /**
   * @brief Méthode permettant d'obtenir la taille d'une colonne.
   * Incluant les hypotétiques 'trous' dans la colonne.
   * 
   * @param pos Le nom de la colonne.
   * @return Integer La taille de la colonne (0 si non trouvée).
   */
  virtual Integer sizeColumn(const String& name_column) = 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant de récupérer la position d'une ligne.
   * 
   * @param name_row Le nom de la ligne.
   * @return Integer La position de la ligne (-1 si non trouvée).
   */
  virtual Integer posRow(const String& name_row) = 0;
  /**
   * @brief Méthode permettant de récupérer la position d'une colonne.
   * 
   * @param name_row Le nom de la colonne.
   * @return Integer La position de la colonne (-1 si non trouvée).
   */
  virtual Integer posColumn(const String& name_column) = 0;
  
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant de récupérer le nombre de lignes dans le tableau.
   * C'est, en quelque sorte, le nombre max d'élements que peut contenir une colonne.
   * 
   * @return Integer Le nombre de lignes du tableau.
   */
  virtual Integer numRows() = 0;
  /**
   * @brief Méthode permettant de récupérer le nombre de colonnes dans le tableau.
   * C'est, en quelque sorte, le nombre max d'élements que peut contenir une ligne.
   * 
   * @return Integer Le nombre de colonnes du tableau.
   */
  virtual Integer numColumns() = 0;
  
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

  virtual String nameRow(Integer pos) = 0;
  virtual String nameColumn(Integer pos) = 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant de changer le nom d'une ligne.
   * 
   * @param pos La position de la ligne.
   * @param new_name Le nouveau nom de la ligne.
   * @return true Si le changement a eu lieu.
   * @return false Si le changement n'a pas eu lieu.
   */
  virtual bool editNameRow(Integer pos, const String& new_name) = 0;
  /**
   * @brief Méthode permettant de changer le nom d'une ligne.
   * 
   * @param name_row Le nom actuel de la ligne.
   * @param new_name Le nouveau nom de la ligne.
   * @return true Si le changement a eu lieu.
   * @return false Si le changement n'a pas eu lieu.
   */
  virtual bool editNameRow(const String& name_row, const String& new_name) = 0;

  /**
   * @brief Méthode permettant de changer le nom d'une colonne.
   * 
   * @param pos La position de la colonne.
   * @param new_name Le nouveau nom de la colonne.
   * @return true Si le changement a eu lieu.
   * @return false Si le changement n'a pas eu lieu.
   */
  virtual bool editNameColumn(Integer pos, const String& new_name) = 0;
  /**
   * @brief Méthode permettant de changer le nom d'une colonne.
   * 
   * @param name_column Le nom actuel de la colonne.
   * @param new_name Le nouveau nom de la colonne.
   * @return true Si le changement a eu lieu.
   * @return false Si le changement n'a pas eu lieu.
   */
  virtual bool editNameColumn(const String& name_column, const String& new_name) = 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant de créer une colonne contenant la moyenne des
   * éléments de chaque ligne.
   * 
   * @param name_column Le nom de la nouvelle colonne.
   * @return Integer La position de la colonne.
   */
  virtual Integer addAverageColumn(const String& name_column) = 0;

  /**
   * @brief Méthode permettant de récupérer le pointeur vers l'objet
   * SimpleTableInternal utilisé.
   * 
   * @return SimpleTableInternal* Le pointeur utilisé. 
   */
  virtual SimpleTableInternal* internal() = 0;

  /**
   * @brief Méthode permettant de définir un pointeur vers
   * SimpleTableInternal.
   * 
   * @warning Il est déconseillé d'utiliser cette méthode, sauf si
   * vous savez ce que vous faite. La destruction de l'objet reste
   * à la charge de l'appelant.
   * 
   * @param strw Le pointeur vers SimpleTableInternal.
   */
  virtual void setInternal(SimpleTableInternal* sti) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
