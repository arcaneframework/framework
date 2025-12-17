// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ISimpleTableInternalMng.h                                   (C) 2000-2025 */
/*                                                                           */
/* Interface représentant un gestionnaire de SimpleTableInternal.            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ISIMPLETABLEINTERNALMNG_H
#define ARCANE_CORE_ISIMPLETABLEINTERNALMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/Array2.h"

#include "arcane/core/SimpleTableInternal.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
 * Ce pointeur peut être placé n'importe où grâce au méthodes element().
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
   * @param row_name Le nom de la ligne. Doit être non vide.
   * @return Integer La position de la ligne dans le tableau 
   *                 (-1 si le nom donné est incorrect).
   */
  virtual Integer addRow(const String& row_name) = 0;
  /**
   * @brief Méthode permettant d'ajouter une ligne.
   * 
   * Si le nombre d'élements dans 'elements' est plus grand que le
   * nombre de colonnes, l'ajout s'effectue quand même (mais les
   * éléments en trop ne seront pas ajoutés).
   * 
   * @param row_name Le nom de la ligne. Doit être non vide.
   * @param elements Les éléments à insérer sur la ligne.
   * @return Integer La position de la ligne dans le tableau.
   *                 (-1 si le nom donné est incorrect).
   */
  virtual Integer addRow(const String& row_name, ConstArrayView<Real> elements) = 0;
  /**
   * @brief Méthode permettant d'ajouter plusieurs lignes.
   * 
   * @param rows_names Les noms des lignes. Chaque nom doit être non vide.
   * @return true Si toutes les lignes ont été créées.
   * @return false Si toutes les lignes n'ont pas été créées.
   */
  virtual bool addRows(StringConstArrayView rows_names) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant d'ajouter une colonne.
   * 
   * @param column_name Le nom de la colonne. Doit être non vide.
   * @return Integer La position de la colonne dans le tableau.
   *                 (-1 si le nom donné est incorrect).
   */
  virtual Integer addColumn(const String& column_name) = 0;
  /**
   * @brief Méthode permettant d'ajouter une colonne.
   * 
   * Si le nombre d'élements dans 'elements' est plus grand que le
   * nombre de lignes, l'ajout s'effectue quand même (mais les
   * éléments en trop ne seront pas ajoutés).
   * 
   * @param column_name Le nom de la colonne. Doit être non vide.
   * @param elements Les éléments à ajouter sur la colonne.
   * @return Integer La position de la colonne dans le tableau.
   *                 (-1 si le nom donné est incorrect).
   */
  virtual Integer addColumn(const String& column_name, ConstArrayView<Real> elements) = 0;
  /**
   * @brief Méthode permettant d'ajouter plusieurs colonnes.
   * 
   * @param rows_names Les noms des colonnes. Chaque nom doit être non vide.
   * @return true Si toutes les colonnes ont été créées.
   * @return false Si toutes les colonnes n'ont pas été créées.
   */
  virtual bool addColumns(StringConstArrayView columns_names) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant d'ajouter un élément à une ligne.
   * 
   * @param position La position de la ligne.
   * @param element L'élément à ajouter.
   * @return true Si l'élément a pu être ajouté.
   * @return false Si l'élément n'a pas pu être ajouté.
   */
  virtual bool addElementInRow(Integer position, Real element) = 0;
  /**
   * @brief Méthode permettant l'ajouter un élément sur une ligne.
   * 
   * @param row_name Le nom de la ligne.
   * @param element L'élément à ajouter.
   * @param create_if_not_exist Pour savoir si l'on doit créer la 
   *                            ligne si elle n'existe pas encore.
   * @return true Si l'élément a pu être ajouté.
   * @return false Si l'élément n'a pas pu être ajouté.
   */
  virtual bool addElementInRow(const String& row_name, Real element, bool create_if_not_exist = true) = 0;
  /**
   * @brief Méthode permettant d'ajouter un élément sur la ligne 
   * dernièrement manipulée.
   * 
   * Cette méthode diffère de 'editElementRight()' car ici, on ajoute 
   * un élément à la fin de la ligne, pas forcement après le
   * dernier élement ajouté.
   * 
   * @param element L'élément à ajouter.
   * @return true Si l'élément a été ajouté.
   * @return false Si l'élément n'a pas pu être ajouté.
   */
  virtual bool addElementInSameRow(Real element) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant d'ajouter plusieurs éléments sur une ligne.
   * 
   * Si le nombre d'élements dans 'elements' est plus grand que le
   * nombre de colonnes disponibles, l'ajout s'effectue quand même (mais les
   * éléments en trop ne seront pas ajoutés) et on aura un return false.
   * 
   * @param position La position de la ligne.
   * @param elements Le tableau d'élement à ajouter.
   * @return true Si tous les éléments ont été ajoutés.
   * @return false Si [0;len(elements)[ éléments ont été ajoutés.
   */
  virtual bool addElementsInRow(Integer position, ConstArrayView<Real> elements) = 0;
  /**
   * @brief Méthode permettant d'ajouter plusieurs éléments sur une ligne.
   * 
   * Si le nombre d'élements dans 'elements' est plus grand que le
   * nombre de colonnes disponibles, l'ajout s'effectue quand même (mais les
   * éléments en trop ne seront pas ajoutés) et on aura un return false.
   * 
   * @param row_name Le nom de la ligne.
   * @param elements Le tableau d'élement à ajouter.
   * @param create_if_not_exist Pour savoir si l'on doit créer la ligne
   *                            si elle n'existe pas encore.
   * @return true Si tous les éléments ont été ajoutés.
   * @return false Si [0;len(elements)[ éléments ont été ajoutés.
   */
  virtual bool addElementsInRow(const String& row_name, ConstArrayView<Real> elements, bool create_if_not_exist = true) = 0;
  /**
   * @brief Méthode permettant d'ajouter plusieurs éléments sur la 
   * ligne dernièrement manipulée.
   * 
   * Si le nombre d'élements dans 'elements' est plus grand que le
   * nombre de colonnes disponibles, l'ajout s'effectue quand même (mais les
   * éléments en trop ne seront pas ajoutés) et on aura un return false.
   * 
   * Mis à part le fait qu'ici, on manipule un tableau, cette méthode diffère
   * de 'editElementRight()' car ici, on ajoute des éléments à la fin de la ligne,
   * pas forcement après le dernier élement ajouté.
   * 
   * @param elements Le tableau d'élement à ajouter.
   * @return true Si tous les éléments ont été ajoutés.
   * @return false Si [0;len(elements)[ éléments ont été ajoutés.
   */
  virtual bool addElementsInSameRow(ConstArrayView<Real> elements) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant d'ajouter un élément à une colonne.
   * 
   * @param position La position de la colonne.
   * @param element L'élément à ajouter.
   * @return true Si l'élément a pu être ajouté.
   * @return false Si l'élément n'a pas pu être ajouté.
   */
  virtual bool addElementInColumn(Integer position, Real element) = 0;
  /**
   * @brief Méthode permettant l'ajouter un élément sur une colonne.
   * 
   * @param column_name Le nom de la colonne.
   * @param element L'élément à ajouter.
   * @param create_if_not_exist Pour savoir si l'on doit créer la colonne
   *                            si elle n'existe pas encore.
   * @return true Si l'élément a pu être ajouté.
   * @return false Si l'élément n'a pas pu être ajouté.
   */
  virtual bool addElementInColumn(const String& column_name, Real element, bool create_if_not_exist = true) = 0;
  /**
   * @brief Méthode permettant d'ajouter un élément sur la colonne
   * dernièrement manipulée.
   * 
   * Cette méthode diffère de 'editElementDown()' car ici, on ajoute un élément
   * à la fin de la colonne, pas forcement après le dernier élement ajouté.
   * 
   * @param element L'élément à ajouter.
   * @return true Si l'élément a été ajouté.
   * @return false Si l'élément n'a pas pu être ajouté.
   */
  virtual bool addElementInSameColumn(Real element) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant d'ajouter plusieurs éléments sur une colonne.
   * 
   * Si le nombre d'élements dans 'elements' est plus grand que le
   * nombre de lignes disponibles, l'ajout s'effectue quand même (mais les
   * éléments en trop ne seront pas ajoutés) et on aura un return false.
   * 
   * @param position La position de la colonne.
   * @param elements Le tableau d'élement à ajouter.
   * @return true Si tous les éléments ont été ajoutés.
   * @return false Si [0;len(elements)[ éléments ont été ajoutés.
   */
  virtual bool addElementsInColumn(Integer position, ConstArrayView<Real> elements) = 0;
  /**
   * @brief Méthode permettant d'ajouter plusieurs éléments sur une colonne.
   * 
   * Si le nombre d'élements dans 'elements' est plus grand que le
   * nombre de lignes disponibles, l'ajout s'effectue quand même (mais les
   * éléments en trop ne seront pas ajoutés) et on aura un return false.
   * 
   * @param column_name Le nom de la colonne.
   * @param elements Le tableau d'élement à ajouter.
   * @param create_if_not_exist Pour savoir si l'on doit créer la colonne si
   *                            elle n'existe pas encore.
   * @return true Si tous les éléments ont été ajoutés.
   * @return false Si [0;len(elements)[ éléments ont été ajoutés.
   */
  virtual bool addElementsInColumn(const String& column_name, ConstArrayView<Real> elements, bool create_if_not_exist = true) = 0;
  /**
   * @brief Méthode permettant d'ajouter plusieurs éléments sur la
   * colonne dernièrement manipulée.
   * 
   * Si le nombre d'élements dans 'elements' est plus grand que le
   * nombre de lignes disponibles, l'ajout s'effectue quand même (mais les
   * éléments en trop ne seront pas ajoutés) et on aura un return false.
   * 
   * Mis à part le fait qu'ici, on manipule un tableau, cette méthode diffère
   * de 'editElementDown()' car ici, on ajoute des éléments à la fin de la colonne,
   * pas forcement après le dernier élement ajouté.
   * 
   * @param elements Le tableau d'élement à ajouter.
   * @return true Si tous les éléments ont été ajoutés.
   * @return false Si [0;len(elements)[ éléments ont été ajoutés.
   */
  virtual bool addElementsInSameColumn(ConstArrayView<Real> elements) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant d'éditer un élément au-dessus du dernier
   * élement dernièrement manipulé (ligne du dessus/même colonne).
   * 
   * L'élement que l'on modifie devient donc le dernier élement modifié
   * à la fin de cette méthode (si update_last_position = true).
   * 
   * @param element L'élement à modifier.
   * @param update_last_position Doit-on déplacer le curseur "dernier élement modifié" ?
   * @return true Si l'élément a été modifié.
   * @return false Si l'élément n'a pas pu être modifié.
   */
  virtual bool editElementUp(Real element, bool update_last_position = true) = 0;
  /**
   * @brief Méthode permettant d'éditer un élément en-dessous du dernier 
   * élement dernièrement manipulé (ligne du dessous/même colonne).
   * 
   * L'élement que l'on modifie devient donc le dernier élement modifié 
   * à la fin de cette méthode (si update_last_position = true).
   * 
   * Cette méthode diffère de 'addElementInSameColumn()' car ici, on ajoute 
   * (ou modifie) un élement sous le dernier élement manipulé, qui n'est
   * pas forcement à la fin de la colonne.
   * 
   * @param element L'élement à modifier.
   * @param update_last_position Doit-on déplacer le curseur "dernier élement modifié" ?
   * @return true Si l'élément a été modifié.
   * @return false Si l'élément n'a pas pu être modifié.
   */
  virtual bool editElementDown(Real element, bool update_last_position = true) = 0;
  /**
   * @brief Méthode permettant d'éditer un élément à gauche du dernier
   * élement dernièrement manipulé (même ligne/colonne à gauche).
   * 
   * L'élement que l'on modifie devient donc le dernier élement modifié
   * à la fin de cette méthode (si update_last_position = true).
   * 
   * @param element L'élement à modifier.
   * @param update_last_position Doit-on déplacer le curseur "dernier élement modifié" ?
   * @return true Si l'élément a été modifié.
   * @return false Si l'élément n'a pas pu être modifié.
   */
  virtual bool editElementLeft(Real element, bool update_last_position = true) = 0;
  /**
   * @brief Méthode permettant d'éditer un élément à droite du dernier
   * élement dernièrement manipulé (même ligne/colonne à droite).
   * 
   * L'élement que l'on modifie devient donc le dernier élement modifié
   * à la fin de cette méthode (si update_last_position = true).
   * 
   * Cette méthode diffère de 'addElementInSameRow()' car ici, on ajoute 
   * (ou modifie) un élement à la droite du dernier élement manipulé, 
   * qui n'est pas forcement à la fin de la colonne.
   * 
   * @param element L'élement à modifier.
   * @param update_last_position Doit-on déplacer le curseur "dernier élement modifié" ?
   * @return true Si l'élément a été modifié.
   * @return false Si l'élément n'a pas pu être modifié.
   */
  virtual bool editElementRight(Real element, bool update_last_position = true) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant de récupérer un élément au-dessus du dernier
   * élement dernièrement manipulé (ligne du dessus/même colonne).
   * 
   * L'élement que l'on récupère devient donc le dernier élement "modifié"
   * à la fin de cette méthode (si update_last_position = true).
   * 
   * @param update_last_position Doit-on déplacer le curseur "dernier élement modifié" ?
   * @return Real L'élement trouvé (0 si non trouvé).
   */
  virtual Real elementUp(bool update_last_position = false) = 0;
  /**
   * @brief Méthode permettant de récupérer un élément en-dessous du dernier
   * élement dernièrement manipulé (ligne du dessous/même colonne).
   * 
   * L'élement que l'on récupère devient donc le dernier élement "modifié"
   * à la fin de cette méthode (si update_last_position = true).
   * 
   * @param update_last_position Doit-on déplacer le curseur "dernier élement modifié" ?
   * @return Real L'élement trouvé (0 si non trouvé).
   */
  virtual Real elementDown(bool update_last_position = false) = 0;
  /**
   * @brief Méthode permettant de récupérer un élément à gauche du dernier
   * élement dernièrement manipulé (même ligne/colonne à gauche).
   * 
   * L'élement que l'on récupère devient donc le dernier élement "modifié"
   * à la fin de cette méthode (si update_last_position = true).
   * 
   * @param update_last_position Doit-on déplacer le curseur "dernier élement modifié" ?
   * @return Real L'élement trouvé (0 si non trouvé).
   */
  virtual Real elementLeft(bool update_last_position = false) = 0;
  /**
   * @brief Méthode permettant de récupérer un élément à droite du dernier
   * élement dernièrement manipulé (même ligne/colonne à droite).
   * 
   * L'élement que l'on récupère devient donc le dernier élement "modifié"
   * à la fin de cette méthode (si update_last_position = true).
   * 
   * @param update_last_position Doit-on déplacer le curseur "dernier élement modifié" ?
   * @return Real L'élement trouvé (0 si non trouvé).
   */
  virtual Real elementRight(bool update_last_position = false) = 0;

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
   * @param element L'élement remplaçant.
   * @return true Si l'élement a bien été remplacé.
   * @return false Si l'élement n'a pas été remplacé.
   */
  virtual bool editElement(Real element) = 0;
  /**
   * @brief Méthode permettant de modifier un élement dans le tableau.
   * 
   * @param position_x La position de la colonne à modifier.
   * @param position_y La position de la ligne à modifier.
   * @param element L'élement remplaçant.
   * @return true Si l'élement a bien été remplacé.
   * @return false Si l'élement n'a pas été remplacé.
   */
  virtual bool editElement(Integer position_x, Integer position_y, Real element) = 0;
  /**
   * @brief Méthode permettant de modifier un élement dans le tableau.
   * 
   * @param column_name Le nom de la colonne où se trouve l'élement.
   * @param row_name Le nom de la ligne où se trouve l'élement.
   * @param element L'élement remplaçant.
   * @return true Si l'élement a bien été remplacé.
   * @return false Si l'élement n'a pas pu être remplacé.
   */
  virtual bool editElement(const String& column_name, const String& row_name, Real element) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant d'obtenir une copie d'un élement.
   * 
   * Les positions x et y correspondent à l'emplacement du dernier élement manipulé.
   * 
   * @return Real L'élement trouvé (0 si non trouvé).
   */
  virtual Real element() = 0;
  /**
   * @brief Méthode permettant d'obtenir une copie d'un élement.
   * 
   * @param position_x La position de la colonne où se trouve l'élement.
   * @param position_y La position de la ligne où se trouve l'élement.
   * @param update_last_position Doit-on déplacer le curseur "dernier élement modifié" ?
   * @return Real L'élement trouvé (0 si non trouvé).
   */
  virtual Real element(Integer position_x, Integer position_y, bool update_last_position = false) = 0;
  /**
   * @brief Méthode permettant d'obtenir une copie d'un élement.
   * 
   * @param column_name Le nom de la colonne où se trouve l'élement.
   * @param row_name Le nom de la ligne où se trouve l'élement.
   * @param update_last_position Doit-on déplacer le curseur "dernier élement modifié" ?
   * @return Real L'élement trouvé (0 si non trouvé).
   */
  virtual Real element(const String& column_name, const String& row_name, bool update_last_position = false) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant d'obtenir une copie d'une ligne.
   * 
   * @param position La position de la ligne.
   * @return RealUniqueArray La copie de la ligne (tableau vide si non trouvée).
   */
  virtual RealUniqueArray row(Integer position) = 0;
  /**
   * @brief Méthode permettant d'obtenir une copie d'une ligne.
   * 
   * @param row_name Le nom de la ligne.
   * @return RealUniqueArray La copie de la ligne (tableau vide si non trouvée).
   */
  virtual RealUniqueArray row(const String& row_name) = 0;

  /**
   * @brief Méthode permettant d'obtenir une copie d'une colonne.
   * 
   * @param position La position de la colonne.
   * @return RealUniqueArray La copie de la colonne (tableau vide si non trouvée).
   */
  virtual RealUniqueArray column(Integer position) = 0;
  /**
   * @brief Méthode permettant d'obtenir une copie d'une colonne.
   * 
   * @param column_name Le nom de la colonne.
   * @return RealUniqueArray La copie de la colonne (tableau vide si non trouvée).
   */
  virtual RealUniqueArray column(const String& column_name) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant d'obtenir la taille d'une ligne.
   * Incluant les hypothétiques 'trous' dans la ligne.
   * 
   * @param position La position de la ligne.
   * @return Integer La taille de la ligne (0 si non trouvée).
   */
  virtual Integer rowSize(Integer position) = 0;
  /**
   * @brief Méthode permettant d'obtenir la taille d'une ligne.
   * Incluant les hypotétiques 'trous' dans la ligne.
   * 
   * @param position Le nom de la ligne.
   * @return Integer La taille de la ligne (0 si non trouvée).
   */
  virtual Integer rowSize(const String& row_name) = 0;

  /**
   * @brief Méthode permettant d'obtenir la taille d'une colonne.
   * Incluant les hypotétiques 'trous' dans la colonne.
   * 
   * @param position La position de la colonne.
   * @return Integer La taille de la colonne (0 si non trouvée).
   */
  virtual Integer columnSize(Integer position) = 0;
  /**
   * @brief Méthode permettant d'obtenir la taille d'une colonne.
   * Incluant les hypotétiques 'trous' dans la colonne.
   * 
   * @param position Le nom de la colonne.
   * @return Integer La taille de la colonne (0 si non trouvée).
   */
  virtual Integer columnSize(const String& column_name) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant de récupérer la position d'une ligne.
   * 
   * @param row_name Le nom de la ligne.
   * @return Integer La position de la ligne (-1 si non trouvée).
   */
  virtual Integer rowPosition(const String& row_name) = 0;
  /**
   * @brief Méthode permettant de récupérer la position d'une colonne.
   * 
   * @param row_name Le nom de la colonne.
   * @return Integer La position de la colonne (-1 si non trouvée).
   */
  virtual Integer columnPosition(const String& column_name) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant de récupérer le nombre de lignes dans le tableau.
   * C'est, en quelque sorte, le nombre max d'élements que peut contenir une colonne.
   * 
   * @return Integer Le nombre de lignes du tableau.
   */
  virtual Integer numberOfRows() = 0;
  /**
   * @brief Méthode permettant de récupérer le nombre de colonnes dans le tableau.
   * C'est, en quelque sorte, le nombre max d'élements que peut contenir une ligne.
   * 
   * @return Integer Le nombre de colonnes du tableau.
   */
  virtual Integer numberOfColumns() = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant de récupérer le nom d'une ligne
   * à partir de sa position.
   * 
   * @param position La position de la ligne.
   * @return String Le nom de la ligne 
   *         (chaine vide si la ligne n'a pas été trouvé).
   */
  virtual String rowName(Integer position) = 0;

  /**
   * @brief Méthode permettant de récupérer le nom d'une colonne
   * à partir de sa position.
   * 
   * @param position La position de la colonne.
   * @return String Le nom de la colonne 
   *         (chaine vide si la colonne n'a pas été trouvé).
   */
  virtual String columnName(Integer position) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant de changer le nom d'une ligne.
   * 
   * @param position La position de la ligne.
   * @param new_name Le nouveau nom de la ligne. Doit être non vide.
   * @return true Si le changement a eu lieu.
   * @return false Si le changement n'a pas eu lieu.
   */
  virtual bool editRowName(Integer position, const String& new_name) = 0;
  /**
   * @brief Méthode permettant de changer le nom d'une ligne.
   * 
   * @param row_name Le nom actuel de la ligne.
   * @param new_name Le nouveau nom de la ligne. Doit être non vide.
   * @return true Si le changement a eu lieu.
   * @return false Si le changement n'a pas eu lieu.
   */
  virtual bool editRowName(const String& row_name, const String& new_name) = 0;

  /**
   * @brief Méthode permettant de changer le nom d'une colonne.
   * 
   * @param position La position de la colonne.
   * @param new_name Le nouveau nom de la colonne. Doit être non vide.
   * @return true Si le changement a eu lieu.
   * @return false Si le changement n'a pas eu lieu.
   */
  virtual bool editColumnName(Integer position, const String& new_name) = 0;
  /**
   * @brief Méthode permettant de changer le nom d'une colonne.
   * 
   * @param column_name Le nom actuel de la colonne.
   * @param new_name Le nouveau nom de la colonne. Doit être non vide.
   * @return true Si le changement a eu lieu.
   * @return false Si le changement n'a pas eu lieu.
   */
  virtual bool editColumnName(const String& column_name, const String& new_name) = 0;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant de créer une colonne contenant la moyenne des
   * éléments de chaque ligne.
   * 
   * @param column_name Le nom de la nouvelle colonne. Doit être non vide.
   * @return Integer La position de la colonne.
   */
  virtual Integer addAverageColumn(const String& column_name) = 0;

  /**
   * @brief Méthode permettant de récupérer une référence vers l'objet
   * SimpleTableInternal utilisé.
   * 
   * @return Ref<SimpleTableInternal> Une copie de la référence. 
   */
  virtual Ref<SimpleTableInternal> internal() = 0;

  /**
   * @brief Méthode permettant de définir une référence vers un
   * SimpleTableInternal.
   * 
   * @param simple_table_internal La référence vers un SimpleTableInternal.
   */
  virtual void setInternal(const Ref<SimpleTableInternal>& simple_table_internal) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
