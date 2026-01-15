// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AMRPatchPosition.h                                          (C) 2000-2025 */
/*                                                                           */
/* Position d'un patch AMR d'un maillage cartésien.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_AMRPATCHPOSITION_H
#define ARCANE_CARTESIANMESH_AMRPATCHPOSITION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/CartesianMeshGlobal.h"
#include "arcane/utils/Vector3.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Classe permettant de définir la position d'un patch dans le maillage
 * cartésien.
 *
 * La position d'un patch est désigné par la position de deux mailles dans la
 * grille. La position "min" et la position "max" forment une boite englobante.
 *
 * \warning La maille à la position "min" est incluse dans la boite, mais la
 * maille à la position "max" est exclue.
 *
 * \note La position du patch est globale pour le maillage cartesien. Le
 * découpage en sous-domaine n'est pas pris en compte (exemple avec la méthode
 * \a nbCells() de cette classe qui donne le nombre de mailles du patch sans
 * tenir compte des sous-domaines).
 *
 * Les positions des mailles peuvent être obtenues par le
 * CartesianMeshNumberingMng.
 *
 * \warning Cette classe est valide uniquement pour un pattern de raffinement de
 * 2 (modifier ça ne devrait pas être complexe, si besoin).
 */
class ARCANE_CARTESIANMESH_EXPORT AMRPatchPosition
{
 public:

  /*!
   * \brief Constructeur pour une position nulle.
   * Une position nulle est définie par un level = -2.
   */
  AMRPatchPosition();
  AMRPatchPosition(Int32 level, CartCoord3 min_point, CartCoord3 max_point, Int32 overlap_layer_size);

  /*!
   * \brief Constructeur de copie.
   * \param src La position à copier.
   */
  AMRPatchPosition(const AMRPatchPosition& src);
  AMRPatchPosition& operator=(const AMRPatchPosition&) = default;

  ~AMRPatchPosition();

 public:

  bool operator==(const AMRPatchPosition& other) const = default;

 public:

  /*!
   * \brief Méthode permettant de récupérer le niveau du patch.
   * \return Le niveau du patch.
   */
  Int32 level() const;

  /*!
   * \brief Méthode permettant de définir le niveau du patch.
   * \param level Le niveau du patch.
   */
  void setLevel(Int32 level);

  /*!
   * \brief Méthode permettant de récupérer la position min de la boite
   * englobante.
   *
   * \return La position min.
   */
  CartCoord3 minPoint() const;

  /*!
   * \brief Méthode permettant de définir la position min de la boite
   * englobante.
   * \param min_point la position min.
   */
  void setMinPoint(CartCoord3 min_point);

  /*!
   * \brief Méthode permettant de récupérer la position max de la boite
   * englobante.
   *
   * \return La position max.
   */
  CartCoord3 maxPoint() const;

  /*!
   * \brief Méthode permettant de définir la position max de la boite
   * englobante.
   * \param max_point la position max.
   */
  void setMaxPoint(CartCoord3 max_point);

  /*!
   * \brief Méthode permettant de récupérer le nombre de couches de mailles de
   * recouvrement du patch.
   *
   * \return le nombre de couches de mailles de recouvrement
   */
  Int32 overlapLayerSize() const;

  /*!
   * \brief Méthode permettant de définir le nombre de couches de mailles de
   * recouvrement du patch.
   * \param layer_size le nombre de couches de mailles de recouvrement
   */
  void setOverlapLayerSize(Int32 layer_size);

  /*!
   * \brief Méthode permettant de récupérer la position min de la boite
   * englobante en incluant la couche de mailles de recouvrement.
   * \return La position min avec la couche de mailles de recouvrement.
   */
  CartCoord3 minPointWithOverlap() const;

  /*!
   * \brief Méthode permettant de récupérer la position max de la boite
   * englobante en incluant la couche de mailles de recouvrement.
   * \return La position max avec la couche de mailles de recouvrement.
   */
  CartCoord3 maxPointWithOverlap() const;

  /*!
   * \brief Méthode permettant de connaitre le nombre de mailles du patch
   * selon sa position.
   *
   * \warning Le nombre de mailles est calculé avec les positions min et max
   * (sans la couche de recouvrement). Ce nombre est donc le même pour tous
   * les sous-domaines. Attention à ne pas comparer ce nombre avec le nombre
   * de mailles du groupe de mailles qui peut être associé à cette classe et
   * qui peut être différent pour chaque sous-domaine.
   *
   * \return Le nombre de maille du patch.
   */
  Int64 nbCells() const;

  /*!
   * \brief Méthode permettant de découper le patch en deux patchs selon un
   * point de découpe.
   *
   * \param cut_point Le point de découpe.
   * \param dim La dimension qui doit être découpée.
   * \return Les deux positions de patch résultant de la découpe.
   */
  std::pair<AMRPatchPosition, AMRPatchPosition> cut(CartCoord cut_point, Integer dim) const;

  /*!
   * \brief Méthode permettant de savoir si notre patch peut être fusionné
   * avec \a other_patch.
   *
   * \param other_patch Le patch à verifier.
   * \return True si la fusion est possible.
   */
  bool canBeFusion(const AMRPatchPosition& other_patch) const;

  /*!
   * \brief Méthode permettant de fusionner \a other_patch avec le nôtre.
   *
   * Une vérification de possibilité de fusion (via \a canBeFusion()) est
   * réalisée avant de fusionner. Si la fusion est impossible, on retourne
   * false. Sinon, on fusionne et on retourne true.
   * Si fusion, \a other_patch devient null.
   *
   * \param other_patch Le patch avec lequel fusionner.
   * \return true si la fusion à été réalisé, false si la fusion est
   * impossible.
   */
  bool fusion(AMRPatchPosition& other_patch);

  /*!
   * \brief Méthode permettant de savoir si la position du patch est nulle.
   *
   * \warning On ne vérifie pas la validité de la position.
   *
   * \return True si le patch est nulle.
   */
  bool isNull() const;

  /*!
   * \brief Méthode permettant de créer un \a AMRPatchPosition pour le niveau
   * supérieur.
   *
   * \param dim La dimension du maillage.
   * \param higher_level Le plus haut niveau de raffinement du maillage.
   * \param overlap_layer_size_top_level Le nombre de couches de mailles de
   * recouvrement pour les patchs du plus haut niveau de raffinement.
   * \return Un \a AMRPatchPosition de niveau supérieur.
   */
  AMRPatchPosition patchUp(Integer dim, Int32 higher_level, Int32 overlap_layer_size_top_level) const;

  /*!
   * \brief Méthode permettant de créer un \a AMRPatchPosition pour le niveau
   * inférieur.
   *
   * Si la position min n'est pas divisible par deux, on arrondit à l'entier
   * inférieur.
   *
   * Si la position max n'est pas divisible par deux, on arrondit à l'entier
   * supérieur.
   *
   * Pour la couche de recouvrement, cette méthode s'assure que l'on n'aura
   * jamais plus d'un niveau de différence entre deux mailles de niveaux
   * différents.
   *
   * \warning patch.patchDown(patch.patchUp(X)) != patch et
   * patch.patchUp(patch.patchDown(X)) != patch.
   *
   * \param dim La dimension du maillage.
   * \param higher_level Le plus haut niveau de raffinement du maillage.
   * \param overlap_layer_size_top_level Le nombre de couches de mailles de
   * recouvrement pour les patchs du plus haut niveau de raffinement.
   * \return Un \a AMRPatchPosition de niveau inférieur.
   */
  AMRPatchPosition patchDown(Integer dim, Int32 higher_level, Int32 overlap_layer_size_top_level) const;

  /*!
   * \brief Méthode permettant de connaitre la taille du patch (en nombre de
   * mailles par direction).
   *
   * \return La taille du patch.
   */
  CartCoord3 length() const;

  /*!
   * \brief Méthode permettant de savoir si une maille de position x,y,z est
   * incluse dans ce patch.
   *
   * Pour inclure la couche de recouvrement, utiliser la méthode
   * \a isInWithOverlap().
   *
   * \param x Position X de la maille.
   * \param y Position Y de la maille.
   * \param z Position Z de la maille.
   *
   * \return True si la maille est dans le patch.
   */
  bool isIn(CartCoord x, CartCoord y, CartCoord z) const;

  /*!
   * \brief Méthode permettant de savoir si une maille est incluse dans ce
   * patch.
   *
   * Pour inclure la couche de recouvrement, utiliser la méthode
   * \a isInWithOverlap().
   *
   * \param coord Position de la maille.
   *
   * \return True si la maille est dans le patch.
   */
  bool isIn(CartCoord3 coord) const;

  /*!
   * \brief Méthode permettant de savoir si une maille de position x,y,z est
   * incluse dans ce patch avec couche de recouvrement.
   *
   * \param x Position X de la maille.
   * \param y Position Y de la maille.
   * \param z Position Z de la maille.
   *
   * \return True si la maille est dans le patch.
   */
  bool isInWithOverlap(CartCoord x, CartCoord y, CartCoord z) const;

  /*!
   * \brief Méthode permettant de savoir si une maille est incluse dans ce
   * patch avec couche de recouvrement.
   *
   * \param coord Position de la maille.
   *
   * \return True si la maille est dans le patch.
   */
  bool isInWithOverlap(CartCoord3 coord) const;

  /*!
   * \brief Méthode permettant de savoir si une maille de position x,y,z est
   * incluse dans ce patch avec couche de recouvrement fourni en paramètre.
   *
   * \param x Position X de la maille.
   * \param y Position Y de la maille.
   * \param z Position Z de la maille.
   * \param overlap Le nombre de mailles de recouvrement de la couche.
   *
   * \return True si la maille est dans le patch.
   */
  bool isInWithOverlap(CartCoord x, CartCoord y, CartCoord z, Integer overlap) const;

  /*!
   * \brief Méthode permettant de savoir si une maille est incluse dans ce
   * patch avec couche de recouvrement fourni en paramètre.
   *
   * \param coord Position de la maille.
   * \param overlap Le nombre de mailles de recouvrement de la couche.
   *
   * \return True si la maille est dans le patch.
   */
  bool isInWithOverlap(CartCoord3 coord, Integer overlap) const;

  /*!
   * \brief Méthode permettant de savoir si notre patch est en contact avec le
   * patch \a other.
   *
   * \param other Le patch à verifier.
   * \return True si les patchs sont en contact.
   */
  bool haveIntersection(const AMRPatchPosition& other) const;

  /*!
   * \brief Méthode permettant de calculer le nombre de couches de mailles de
   * recouvrement pour un niveau donné.
   *
   * \param level Le niveau demandé.
   * \param higher_level Le plus haut niveau de raffinement.
   * \param overlap_layer_size_top_level Le nombre de couches pour le plus
   * haut niveau de raffinement.
   * \return Le nombre de couches de mailles de recouvrement pour le niveau
   * demandé.
   */
  static Int32 computeOverlapLayerSize(Int32 level, Int32 higher_level, Int32 overlap_layer_size_top_level);

  /*!
   * \brief Méthode permettant de calculer le nombre de couches de mailles de
   * recouvrement pour notre patch.
   *
   * \param higher_level Le plus haut niveau de raffinement.
   * \param overlap_layer_size_top_level Le nombre de couches pour le plus
   * haut niveau de raffinement.
   */
  void computeOverlapLayerSize(Int32 higher_level, Int32 overlap_layer_size_top_level);

 private:

  Int32 m_level;
  CartCoord3 m_min_point;
  CartCoord3 m_max_point;
  Int32 m_overlap_layer_size;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

