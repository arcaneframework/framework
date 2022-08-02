// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ISimpleTableOutput.hh                                       (C) 2000-2022 */
/*                                                                           */
/* Interface pour simples services de sortie de tableaux de valeurs.         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_ISIMPLETABLEOUTPUT_H
#define ARCANE_ISIMPLETABLEOUTPUT_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <arcane/ISimpleTableOutputMng.h>
#include <arcane/ISimpleTableMng.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/**
 * @ingroup StandardService
 * @brief Interface représentant une sortie de tableau simple.
 */
class ARCANE_CORE_EXPORT ISimpleTableOutput
: public ISimpleTableMng
, public ISimpleTableOutputMng
{
public:
  virtual ~ISimpleTableOutput() = default;

public:
  /**
   * @brief Méthode permettant d'initialiser le tableau.
   */
  virtual bool init() = 0;
  /**
   * @brief Méthode permettant d'initialiser le tableau.
   * 
   * @param name_table Le nom du tableau (et du fichier de sortie).
   */
  virtual bool init(String name_table) = 0;
  /**
   * @brief Méthode permettant d'initialiser le tableau.
   * 
   * @param name_table Le nom du tableau (et du fichier de sortie).
   * @param name_dir Le nom du dossier dans lequel enregistrer les tableaux.
   */
  virtual bool init(String name_table, String name_dir) = 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


  virtual void clear() = 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant d'ajouter une ligne.
   * 
   * @param name_row Le nom de la ligne.
   * @return Integer La position de la ligne dans le tableau.
   */
  virtual Integer addRow(String name_row) = 0;
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
  virtual Integer addRow(String name_row, ConstArrayView<Real> elems) = 0;
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
  virtual Integer addColumn(String name_column) = 0;
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
  virtual Integer addColumn(String name_column, ConstArrayView<Real> elems) = 0;
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
  virtual bool addElemRow(String name_row, Real elem, bool create_if_not_exist = true) = 0;
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
  virtual bool addElemsRow(String name_row, ConstArrayView<Real> elems, bool create_if_not_exist = true) = 0;
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
  virtual bool addElemColumn(String name_column, Real elem, bool create_if_not_exist = true) = 0;
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
  virtual bool addElemsColumn(String name_column, ConstArrayView<Real> elems, bool create_if_not_exist = true) = 0;
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
  virtual bool editElem(String name_column, String name_row, Real elem) = 0;

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
  virtual Real elem(String name_column, String name_row, bool update_last_pos = false) = 0;

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
  virtual RealUniqueArray row(String name_row) = 0;

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
  virtual RealUniqueArray column(String name_column) = 0;

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
  virtual Integer sizeRow(String name_row) = 0;

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
  virtual Integer sizeColumn(String name_column) = 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant de récupérer la position d'une ligne.
   * 
   * @param name_row Le nom de la ligne.
   * @return Integer La position de la ligne (-1 si non trouvée).
   */
  virtual Integer posRow(String name_row) = 0;
  /**
   * @brief Méthode permettant de récupérer la position d'une colonne.
   * 
   * @param name_row Le nom de la colonne.
   * @return Integer La position de la colonne (-1 si non trouvée).
   */
  virtual Integer posColumn(String name_column) = 0;
  
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
  virtual bool editNameRow(Integer pos, String new_name) = 0;
  /**
   * @brief Méthode permettant de changer le nom d'une ligne.
   * 
   * @param name_row Le nom actuel de la ligne.
   * @param new_name Le nouveau nom de la ligne.
   * @return true Si le changement a eu lieu.
   * @return false Si le changement n'a pas eu lieu.
   */
  virtual bool editNameRow(String name_row, String new_name) = 0;

  /**
   * @brief Méthode permettant de changer le nom d'une colonne.
   * 
   * @param pos La position de la colonne.
   * @param new_name Le nouveau nom de la colonne.
   * @return true Si le changement a eu lieu.
   * @return false Si le changement n'a pas eu lieu.
   */
  virtual bool editNameColumn(Integer pos, String new_name) = 0;
  /**
   * @brief Méthode permettant de changer le nom d'une colonne.
   * 
   * @param name_column Le nom actuel de la colonne.
   * @param new_name Le nouveau nom de la colonne.
   * @return true Si le changement a eu lieu.
   * @return false Si le changement n'a pas eu lieu.
   */
  virtual bool editNameColumn(String name_column, String new_name) = 0;


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
  /**
   * @brief Méthode permettant d'afficher le tableau.
   * 
   * @param only_proc L'id du processus devant afficher le tableau (-1 pour 
   *                  signifier "tous les processus").
   */
  virtual void print(Integer only_proc = 0) = 0;

  virtual bool writeFile(Directory root_dir, Integer only_proc) = 0;

  /**
   * @brief Méthode permettant d'écrire le tableau dans un fichier.
   * Si only_proc != -1, les processus autres que P0 retournent true.
   * 
   * @param only_proc L'id du processus devant écrire dans un fichier 
   *                  le tableau (-1 pour signifier "tous les processus").
   * @return true Si le fichier a été correctement écrit.
   * @return false Si le fichier n'a pas été correctement écrit.
   */
  virtual bool writeFile(Integer only_proc = -1) = 0;
  /**
   * @brief Méthode permettant d'écrire le tableau dans un fichier.
   * Si only_proc != -1, les processus autres que P0 retournent true.
   * 
   * @param dir Le répertoire où sera écrit le fichier
   *            . Le chemin final sera "./[output_dir]/csv/[dir]/"
   * @param only_proc L'id du processus devant écrire dans un fichier 
   *                  le tableau (-1 pour signifier "tous les processus").
   * @return true Si le fichier a été correctement écrit.
   * @return false Si le fichier n'a pas été correctement écrit.
   * 
   * @deprecated Utiliser setOutputDir() puis writeFile() à la place.
   */
  virtual bool writeFile(String dir, Integer only_proc = -1) = 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

  /**
   * @brief Méthode permettant de récupérer la précision actuellement
   * utilisée pour l'écriture des valeurs.
   * 
   * @return Integer La précision.
   */
  virtual Integer precision() = 0;
  /**
   * @brief Méthode permettant de modifier la précision du print.
   * 
   * Aussi bien pour la méthode 'print()' que les méthodes 'writeFile()'.
   * 
   * \note Un appel à cette méthode sans le paramètre définira la précision
   * par défaut.
   * 
   * @param precision La nouvelle précision.
   */
  virtual void setPrecision(Integer precision = 6) = 0;

  /**
   * @brief Méthode permettant de savoir si le frag 'std::fixed' est
   * actif ou non pour l'écriture des valeurs.
   * 
   * @return true Si oui.
   * @return false Si non.
   */
  virtual bool fixed() = 0;
  /**
   * @brief Méthode permettant de définir le flag 'std::fixed' ou non.
   * 
   * Aussi bien pour la méthode 'print()' que les méthodes 'writeFile()'.
   * 
   * Ce flag permet de 'forcer' le nombre de chiffre après la virgule à
   * la précision voulu. Par exemple, si l'on a appelé 'setPrecision(4)',
   * et que l'on appelle 'setFixed(true)', le print de '6.1' donnera '6.1000'.
   * 
   * \note Un appel à cette méthode sans le paramètre définira le flag
   * par défaut.
   * 
   * @param fixed Si le flag 'std::fixed' doit être défini ou non.
   */
  virtual void setFixed(bool fixed = true) = 0;


  /**
   * @brief Accesseur permettant de récupérer le nom du répertoire où sera
   * placé les tableaux.
   * 
   * Peut-être différent pour chaque processus (dépendant de l'implémentation).
   * 
   * @return String Le répertoire.
   */
  virtual String outputDir() = 0;
  /**
   * @brief Accesseur permettant de définir le répertoire
   * dans lequel enregistrer les tableaux.
   * 
   * Peut-être différent pour chaque processus (dépendant de l'implémentation).
   * 
   * @param dir Le répertoire.
   */
  virtual void setOutputDir(String dir) = 0;


  /**
   * @brief Accesseur permettant de récupérer le nom des tableaux.
   * 
   * Peut-être différent pour chaque processus (dépendant de l'implémentation).
   * 
   * @return String Le nom.
   */
  virtual String tabName() = 0;
  /**
   * @brief Accesseur permettant de définir le nom du tableau.
   * 
   * Peut-être différent pour chaque processus (dépendant de l'implémentation).
   * 
   * @param name Le nom.
   */
  virtual void setTabName(String name) = 0;

  /**
   * @brief Accesseur permettant de récupérer le nom des fichiers.
   * 
   * Peut-être différent pour chaque processus (dépendant de l'implémentation).
   * 
   * @return String Le nom.
   */
  virtual String fileName() = 0;

  /**
   * @brief Accesseur permettant de récupérer le chemin où sera
   * enregistrés les tableaux. 
   * 
   * En comparaison avec rootPathOutput(), le retour peut être
   * different selon le "dir" et le "name".
   * 
   * @return String Le chemin.
   */
  virtual Directory outputPath() = 0;


  /**
   * @brief Accesseur permettant de récupérer le chemin où l'implémentation
   * enregistre ces tableaux. 
   * 
   * En comparaison avec pathOutput(), le retour ne dépend pas de "dir" ou de "name".
   * 
   * @return String Le chemin.
   */
  virtual Directory rootPath() = 0;

  /**
   * @brief Méthode permettant de savoir si les paramètres actuellement en possession
   * de l'implémentation lui permet d'écrire un fichier par processus.
   * 
   * @return true Si oui, l'implémentation peut écrire un fichier par processus.
   * @return false Sinon, il n'y a qu'un seul fichier qui peut être écrit.
   */
  virtual bool isOneFileByProcsPermited() = 0;

  virtual String outputFileType() = 0;


  virtual SimpleTableInternal* internal() = 0;
  virtual void setInternal(SimpleTableInternal* sti) = 0;
  virtual void setInternal(SimpleTableInternal& sti) = 0;


  virtual ISimpleTableReaderWriter* readerWriter() = 0;
  virtual void setReaderWriter(ISimpleTableReaderWriter* strw) = 0;
  virtual void setReaderWriter(ISimpleTableReaderWriter& strw) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
