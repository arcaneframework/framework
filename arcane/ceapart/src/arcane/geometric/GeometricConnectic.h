// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GeometricConnectic.h                                        (C) 2000-2014 */
/*                                                                           */
/* Connectiques des élements 2D/3D.                            .             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_GEOMETRIC_GEOMETRICCONNECTIC_H
#define ARCANE_GEOMETRIC_GEOMETRICCONNECTIC_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"

#include "arcane/geometric/GeometricGlobal.h"

#ifdef __GNUC__
#define ARCANE_UNUSED_ATTRIBUTE __attribute__((unused))
#else
#define ARCANE_UNUSED_ATTRIBUTE
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
GEOMETRIC_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Structure de connectique locale
 *
 * Ces trois tableaux permettent de repérer les voisins des éléments
 * de base des mailles pour la numérotation locale.
 *
 * Pour une maille quadrangulaire :
 * -----------------------------
 *
 * Définition de NodeConnectic2D
 *
 *	-> le sommet 0 appartient aux faces 0,1,2
 *	-> le sommet 1 appartient aux faces 0,2,4
 *
 *	and so on.... et bien sûr, le tout dans "l'ordre".
 *
 * Définition de FaceConnectic2D
 *
 *	-> la face 0 est constituée des sommets 0,3
 *	-> la face 1 est constituée des sommets 0,4
 *
 *	l'ordre des sommets définissent la direction de la normale
 *	extèrieure à la maille à travers la face.
 *
 * Définition de SVCFaceConnectic2D
 *
 *      -> la face interne a la maille  du sous volume de contrôle est constituée 
 *         du centre, du centre d'une face
 *         Elle intervient positivement sur un noeud et negativement sur un autre
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief	Structures de connectique locale des mailles 
 *
 *	renvoie le num local dans la maille des sommets de la face
 *
 *	les sommets sont numérotés de sorte qu'ils définissent la
 *	normale extérieure à la maille.
 *
 */
struct FaceConnectic
{
 public:

  Integer nbNode() const { return m_nb_node; }
  Integer node(Integer i) const { return m_node[i]; }

 public:

  FaceConnectic()
  : m_nb_node(0)
  {
    m_node[0] = NULL_ITEM_ID;
    m_node[1] = NULL_ITEM_ID;
    m_node[2] = NULL_ITEM_ID;
    m_node[3] = NULL_ITEM_ID;
    m_node[4] = NULL_ITEM_ID;
    m_node[5] = NULL_ITEM_ID;
  }

  FaceConnectic(Integer n0,Integer n1,Integer n2,
                Integer n3,Integer n4,Integer n5)
  : m_nb_node(6)
  {
    m_node[0] = n0; m_node[1] = n1; m_node[2] = n2;
    m_node[3] = n3; m_node[4] = n4; m_node[5] = n5;
  }

  FaceConnectic(Integer n0,Integer n1,Integer n2,Integer n3,Integer n4)
  : m_nb_node(5)
  {
    m_node[0] = n0; m_node[1] = n1; m_node[2] = n2;
    m_node[3] = n3; m_node[4] = n4;
  }

  FaceConnectic(Integer n0,Integer n1,Integer n2,Integer n3)
  : m_nb_node(4)
  {
    m_node[0] = n0; m_node[1] = n1; m_node[2] = n2; m_node[3] = n3;
  }

  FaceConnectic(Integer n0,Integer n1,Integer n2)
  : m_nb_node(3)
  {
    m_node[0] = n0; m_node[1] = n1; m_node[2] = n2;
  }

  FaceConnectic(Integer n0,Integer n1)
  : m_nb_node(2)
  {
    m_node[0] = n0; m_node[1] = n1;
  }
 private:

  Integer m_node[6];
  Integer m_nb_node;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Structures de connectique locale des mailles.
 *
 * renvoie le num local dans la maille des sommets de l'arete
 * et le numéro local des faces de la maille qui s'appuient sur l'arête.
 *
 * la numérotation des faces est telle que la surface définie par
 * (c1,m,c2,bQ) --- où c1 et c2 représentent les milieux de la
 * première et de la seconde face, m le milieu de l'arête, et bQ
 * le barycentre de la maille --- soit orienté positivement de s1
 * vers s2.
 */
struct EdgeConnectic
{
 public :
  Integer nbNode() const { return 2; }

  EdgeConnectic()
  {
    m_node[0] = NULL_ITEM_ID; m_node[1] = NULL_ITEM_ID;
    m_face[0] = NULL_ITEM_ID; m_face[1] = NULL_ITEM_ID;
  }

  EdgeConnectic(Integer n0,Integer n1,Integer f0,Integer f1 )
  {
    m_node[0] = n0; m_node[1] = n1;
    m_face[0] = f0; m_face[1] = f1;
  }
 public:

  inline Integer node(Integer i) const { return m_node[i]; }
  inline Integer face(Integer i) const { return m_face[i]; }

 private:

  Integer m_node[2];
  Integer m_face[2];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief	Structures de connectique locale des mailles 
 *
 *	renvoie le num local dans la maille des 3 aretes et des 3 faces 
 *	connectees au sommet et de la definition du triedre associe (n0,n1,n2,n3)
 */
struct NodeConnectic
{
 public:
  NodeConnectic()
  {
    m_edge[0] = NULL_ITEM_ID; m_edge[1] = NULL_ITEM_ID; m_edge[2] = NULL_ITEM_ID;
    m_face[0] = NULL_ITEM_ID; m_face[1] = NULL_ITEM_ID; m_face[2] = NULL_ITEM_ID;
    m_node[0] = m_node[1] = m_node[2] = m_node[3] = NULL_ITEM_ID;
  }
  // 3D 
  NodeConnectic(Integer e0,Integer e1,Integer e2,
                Integer f0,Integer f1,Integer f2,
                Integer n0, Integer n1, Integer n2, Integer n3)
  {
    m_edge[0] = e0; m_edge[1] = e1; m_edge[2] = e2;
    m_face[0] = f0; m_face[1] = f1; m_face[2] = f2;
    m_node[0] = n0, m_node[1] = n1, m_node[2] = n2, m_node[3] = n3;
  }

  // 2D 
  NodeConnectic(Integer f0,Integer f1,
                Integer n0, Integer n1, Integer n2)
  {
    m_face[0] = f0; m_face[1] = f1;
    m_node[0] = n0, m_node[1] = n1, m_node[2] = n2;
  }

 public:

  inline Integer edge(Integer i) const { return m_edge[i]; }
  inline Integer face(Integer i) const { return m_face[i]; }
  inline Integer node(Integer i) const { return m_node[i]; }
 private:

  Integer m_edge[3];
  Integer m_face[3];
  Integer m_node[4];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Structures de connectique locale des mailles et svc
 *
 * Renvoie la definition d'une face interne (faces des svc(s) non
 * contenues dans les faces de la maille) ainsi que les deux noeuds
 * separes par cette face
 */
struct SVCFaceConnectic
{
 public:

  SVCFaceConnectic()
  :  m_edge(0), m_face1(0), m_face2(0), m_node_pos(0), m_node_neg(0)
  {}

  // 3D 
  SVCFaceConnectic(Integer face1, Integer edge, Integer face2,
                   Integer node_pos, Integer node_neg)
  : m_edge (edge), m_face1(face1), m_face2(face2), m_node_pos(node_pos),
    m_node_neg (node_neg)
  {
  }
  
  // 2D
  SVCFaceConnectic(Integer face1, Integer node_pos, Integer node_neg)
  : m_edge(NULL_ITEM_LOCAL_ID), m_face1(face1),
    m_face2(NULL_ITEM_LOCAL_ID), m_node_pos(node_pos),
    m_node_neg (node_neg)
  {
  }

  inline Integer edge() const { return m_edge; }
  inline Integer firstFace() const { return m_face1; }
  inline Integer secondFace() const { return m_face2; }
  inline Integer positiveNode () const { return m_node_pos;}
  inline Integer negativeNode () const { return m_node_neg;}

 private:

  Integer m_edge;
  Integer m_face1;
  Integer m_face2;
  Integer m_node_pos;
  Integer m_node_neg;
};  

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


// tableaux de constantes pour les hexaèdres
//------------------------------------------

// noeuds de l'hexaèdre
const Integer hexa_node_association[8]={0,1,2,3,4,5,6,7};

// pour chaque noeud, on donne les 3 arêtes et les 3 faces auxquelles il est connecté
const NodeConnectic hexa_node_connectic[8] ARCANE_UNUSED_ATTRIBUTE =
{
   NodeConnectic( /* Edges */ 0,  3,  4, /* Faces */ 0, 1, 2, /* Nodes */ 0, 1, 3, 4),
   NodeConnectic( /* Edges */ 1,  0,  5, /* Faces */ 0, 2, 4, /* Nodes */ 1, 2, 0, 5),
   NodeConnectic( /* Edges */ 2,  1,  6, /* Faces */ 0, 4, 5, /* Nodes */ 2, 3, 1, 6),
   NodeConnectic( /* Edges */ 3,  2,  7, /* Faces */ 0, 5, 1, /* Nodes */ 3, 0, 2, 7),
   NodeConnectic( /* Edges */ 11, 8,  4, /* Faces */ 3, 2, 1, /* Nodes */ 4, 7, 5, 0),
   NodeConnectic( /* Edges */ 8,  9,  5, /* Faces */ 3, 4, 2, /* Nodes */ 5, 4, 6, 1),
   NodeConnectic( /* Edges */ 9,  10, 6, /* Faces */ 3, 5, 4, /* Nodes */ 6, 5, 7, 2),
   NodeConnectic( /* Edges */ 10, 11, 7, /* Faces */ 3, 1, 5, /* Nodes */ 7, 6, 4, 3)
};

// pour chaque arête, on donne les 2 noeuds qui la composent et les 2 faces auxquelles elle est connectée
const EdgeConnectic hexa_edge_connectic[12] ARCANE_UNUSED_ATTRIBUTE =
{
  EdgeConnectic(0,1,0,2),EdgeConnectic(1,2,0,4),EdgeConnectic(2,3,0,5),EdgeConnectic(3,0,0,1),
  EdgeConnectic(0,4,2,1),EdgeConnectic(1,5,4,2),EdgeConnectic(2,6,5,4),EdgeConnectic(3,7,1,5),
  EdgeConnectic(4,5,2,3),EdgeConnectic(5,6,4,3),EdgeConnectic(6,7,5,3),EdgeConnectic(7,4,1,3)
};

// pour chaque face, on donne les noeuds qui la composent
const FaceConnectic hexa_face_connectic[6] ARCANE_UNUSED_ATTRIBUTE =
{
  FaceConnectic(0,3,2,1),FaceConnectic(0,4,7,3),FaceConnectic(0,1,5,4),
  FaceConnectic(4,5,6,7),FaceConnectic(1,2,6,5),FaceConnectic(2,3,7,6)
};
		
// Generated by calcul.rb
const SVCFaceConnectic hexa_svc_face_connectic[12] ARCANE_UNUSED_ATTRIBUTE =
{
 SVCFaceConnectic(4, 1, 0, 1, 2 ),
 SVCFaceConnectic(1, 11, 3, 4, 7 ),
 SVCFaceConnectic(5, 7, 1, 3, 7 ),
 SVCFaceConnectic(3, 9, 4, 5, 6 ),
 SVCFaceConnectic(2, 0, 0, 0, 1 ),
 SVCFaceConnectic(3, 10, 5, 6, 7 ),
 SVCFaceConnectic(4, 6, 5, 2, 6 ),
 SVCFaceConnectic(2, 5, 4, 1, 5 ),
 SVCFaceConnectic(5, 2, 0, 2, 3 ),
 SVCFaceConnectic(1, 4, 2, 0, 4 ),
 SVCFaceConnectic(0, 3, 1, 0, 3 ),
 SVCFaceConnectic(3, 8, 2, 4, 5 )
};

// tableaux de constantes pour les pyramides
//------------------------------------------

const Integer pyra_node_association[8]={0,1,2,3,4,4,4,4};

const NodeConnectic pyra_node_connectic[8] ARCANE_UNUSED_ATTRIBUTE =
{
   NodeConnectic( /* Edges */ 0, 3, 4, /* Faces */ 0, 1, 2, /* Nodes */ 0, 1, 3, 4),
   NodeConnectic( /* Edges */ 1, 0, 5, /* Faces */ 0, 2, 3, /* Nodes */ 1, 2, 0, 4),
   NodeConnectic( /* Edges */ 2, 1, 6, /* Faces */ 0, 3, 4, /* Nodes */ 2, 3, 1, 4),
   NodeConnectic( /* Edges */ 3, 2, 7, /* Faces */ 0, 4, 1, /* Nodes */ 3, 0, 2, 4),
   NodeConnectic( /* Edges */ 11, 8, 4, /* Faces */ 5, 2, 1, /* Nodes */ 4, 1, 0, 3),
   NodeConnectic( /* Edges */ 8, 9, 5, /* Faces */ 5, 3, 2, /* Nodes */ 4, 2, 1, 0),
   NodeConnectic( /* Edges */ 9, 10, 6, /* Faces */ 5, 4, 3, /* Nodes */ 4, 3, 2, 1),
   NodeConnectic( /* Edges */ 10, 11, 7, /* Faces */ 5, 1, 4, /* Nodes */ 4, 0, 3, 2)
};


const EdgeConnectic pyra_edge_connectic[12] ARCANE_UNUSED_ATTRIBUTE =
{
  EdgeConnectic(0,1,0,2),
  EdgeConnectic(1,2,0,3),
  EdgeConnectic(2,3,0,4),
  EdgeConnectic(3,0,0,1),
  EdgeConnectic(0,4,2,1),
  EdgeConnectic(1,4,3,2),
  EdgeConnectic(2,4,4,3),
  EdgeConnectic(3,4,1,4),
  EdgeConnectic(4,4,2,5),
  EdgeConnectic(4,4,3,5),
  EdgeConnectic(4,4,4,5),
  EdgeConnectic(4,4,1,5)
};

const FaceConnectic pyra_face_connectic[6] ARCANE_UNUSED_ATTRIBUTE =
{
  FaceConnectic(0,3,2,1),
  FaceConnectic(0,4,3),
  FaceConnectic(0,1,4),
  FaceConnectic(1,2,4),
  FaceConnectic(2,3,4),
  FaceConnectic(4,4,4)
};

// Generated by calcul.rb and modified to remove null faces
const SVCFaceConnectic pyra_svc_face_connectic[8] ARCANE_UNUSED_ATTRIBUTE =
{
 SVCFaceConnectic(4, 2, 0, 2, 3 ),
 SVCFaceConnectic(2, 0, 0, 0, 1 ),
 SVCFaceConnectic(4, 7, 1, 3, 4 ),
 SVCFaceConnectic(2, 5, 3, 1, 4 ),
 SVCFaceConnectic(3, 1, 0, 1, 2 ),
 SVCFaceConnectic(3, 6, 4, 2, 4 ),
 SVCFaceConnectic(1, 4, 2, 0, 4 ),
 SVCFaceConnectic(0, 3, 1, 0, 3 )
};


// tableaux de constantes pour les pentaèdres
//-------------------------------------------

const Integer penta_node_association[6]={0,1,2,3,4,5};

const NodeConnectic penta_node_connectic[6] ARCANE_UNUSED_ATTRIBUTE =
{
   NodeConnectic( /* Edges */ 0, 2, 3, /* Faces */ 0, 1, 2, /* Nodes */ 0, 1, 2, 3),
   NodeConnectic( /* Edges */ 1, 0, 4, /* Faces */ 0, 2, 4, /* Nodes */ 1, 2, 0, 4),
   NodeConnectic( /* Edges */ 2, 1, 5, /* Faces */ 0, 4, 1, /* Nodes */ 2, 0, 1, 5),
   NodeConnectic( /* Edges */ 8, 6, 3, /* Faces */ 3, 2, 1, /* Nodes */ 3, 5, 4, 0),
   NodeConnectic( /* Edges */ 6, 7, 4, /* Faces */ 3, 4, 2, /* Nodes */ 4, 3, 5, 1),
   NodeConnectic( /* Edges */ 7, 8, 5, /* Faces */ 3, 1, 4, /* Nodes */ 5, 4, 3, 2)
};


const EdgeConnectic penta_edge_connectic[9] ARCANE_UNUSED_ATTRIBUTE =
{
  EdgeConnectic(0,1,0,2),EdgeConnectic(1,2,0,4),EdgeConnectic(2,0,1,0),EdgeConnectic(0,3,1,2),
  EdgeConnectic(1,4,2,4),EdgeConnectic(2,5,4,1),EdgeConnectic(3,4,2,3),EdgeConnectic(4,5,4,3),
  EdgeConnectic(5,3,1,3)
};

const FaceConnectic penta_face_connectic[5] ARCANE_UNUSED_ATTRIBUTE =
{
  FaceConnectic(0,2,1),FaceConnectic(0,3,5,2),FaceConnectic(0,1,4,3),
  FaceConnectic(3,4,5),FaceConnectic(1,2,5,4)
};

// Generated by calcul.rb
const SVCFaceConnectic penta_svc_face_connectic[9] ARCANE_UNUSED_ATTRIBUTE =
{
 SVCFaceConnectic(1, 8, 3, 3, 5 ),
 SVCFaceConnectic(4, 5, 1, 2, 5 ),
 SVCFaceConnectic(0, 2, 1, 0, 2 ),
 SVCFaceConnectic(4, 1, 0, 1, 2 ),
 SVCFaceConnectic(3, 6, 2, 3, 4 ),
 SVCFaceConnectic(2, 4, 4, 1, 4 ),
 SVCFaceConnectic(2, 0, 0, 0, 1 ),
 SVCFaceConnectic(1, 3, 2, 0, 3 ),
 SVCFaceConnectic(3, 7, 4, 4, 5 )
};


// tableaux de constantes pour les tétraèdres
//-------------------------------------------

const Integer tetra_node_association[4]={0,1,2,3};

const NodeConnectic tetra_node_connectic[4] ARCANE_UNUSED_ATTRIBUTE =
{
   NodeConnectic( /* Edges */ 0, 2, 3, /* Faces */ 0, 1, 2, /* Nodes */ 0, 1, 2, 3),
   NodeConnectic( /* Edges */ 1, 0, 4, /* Faces */ 0, 2, 3, /* Nodes */ 1, 2, 0, 3),
   NodeConnectic( /* Edges */ 2, 1, 5, /* Faces */ 0, 3, 1, /* Nodes */ 2, 0, 1, 3),
   NodeConnectic( /* Edges */ 3, 5, 4, /* Faces */ 1, 3, 2, /* Nodes */ 3, 0, 2, 1)
};

const EdgeConnectic tetra_edge_connectic[6] ARCANE_UNUSED_ATTRIBUTE =
{
  EdgeConnectic(0,1,0,2),EdgeConnectic(1,2,0,3),EdgeConnectic(2,0,0,1),EdgeConnectic(0,3,1,2),
  EdgeConnectic(1,3,2,3),EdgeConnectic(2,3,1,3)
};

const FaceConnectic tetra_face_connectic[4] ARCANE_UNUSED_ATTRIBUTE =
{
  FaceConnectic(0,2,1),FaceConnectic(0,3,2),FaceConnectic(0,1,3),
  FaceConnectic(3,1,2)
};

// Generated by calcul.rb
const SVCFaceConnectic tetra_svc_face_connectic[6] ARCANE_UNUSED_ATTRIBUTE =
{
 SVCFaceConnectic(0, 2, 1, 0, 2 ),
 SVCFaceConnectic(2, 4, 3, 1, 3 ),
 SVCFaceConnectic(2, 0, 0, 0, 1 ),
 SVCFaceConnectic(1, 3, 2, 0, 3 ),
 SVCFaceConnectic(3, 5, 1, 2, 3 ),
 SVCFaceConnectic(3, 1, 0, 1, 2 )
};

// tableaux de constantes pour les prismes à base pentagonale
//-----------------------------------------------------------

const Integer wedge7_node_association[10]={0,1,2,3,4,5,6,7,8,9};

const NodeConnectic wedge7_node_connectic[10] ARCANE_UNUSED_ATTRIBUTE =
{
   NodeConnectic( /* Edges */ 0, 4, 10, /* Faces */ 0, 6, 2, /* Nodes */ 0, 1, 4, 5),
   NodeConnectic( /* Edges */ 1, 0, 11, /* Faces */ 0, 2, 3, /* Nodes */ 1, 2, 0, 6),
   NodeConnectic( /* Edges */ 2, 1, 12, /* Faces */ 0, 3, 4, /* Nodes */ 2, 3, 1, 7),
   NodeConnectic( /* Edges */ 3, 2, 13, /* Faces */ 0, 4, 5, /* Nodes */ 3, 4, 2, 8),
   NodeConnectic( /* Edges */ 4, 3, 14, /* Faces */ 0, 5, 6, /* Nodes */ 4, 0, 3, 9),
   NodeConnectic( /* Edges */ 9, 5, 10, /* Faces */ 1, 2, 6, /* Nodes */ 5, 9, 6, 0),
   NodeConnectic( /* Edges */ 5, 6, 11, /* Faces */ 1, 3, 2, /* Nodes */ 6, 5, 7, 1),
   NodeConnectic( /* Edges */ 6, 7, 12, /* Faces */ 1, 4, 3, /* Nodes */ 7, 6, 8, 2),
   NodeConnectic( /* Edges */ 7, 8, 13, /* Faces */ 1, 5, 4, /* Nodes */ 8, 7, 9, 3),
   NodeConnectic( /* Edges */ 8, 9, 14, /* Faces */ 1, 6, 5, /* Nodes */ 9, 8, 5, 4)
};

const EdgeConnectic wedge7_edge_connectic[15] ARCANE_UNUSED_ATTRIBUTE =
{
  EdgeConnectic(0,1,0,2),EdgeConnectic(1,2,0,3),EdgeConnectic(2,3,0,4),EdgeConnectic(3,4,0,5),
  EdgeConnectic(4,0,0,6),EdgeConnectic(5,6,2,1),EdgeConnectic(6,7,3,1),EdgeConnectic(7,8,4,1),
  EdgeConnectic(8,9,5,1),EdgeConnectic(9,5,6,1),EdgeConnectic(0,5,6,2),EdgeConnectic(1,6,2,3),
  EdgeConnectic(2,7,3,4),EdgeConnectic(3,8,4,5),EdgeConnectic(4,9,5,6)
};

const FaceConnectic wedge7_face_connectic[7] ARCANE_UNUSED_ATTRIBUTE =
{
  FaceConnectic(0,4,3,2,1),FaceConnectic(5,6,7,8,9),FaceConnectic(0,1,6,5),
  FaceConnectic(1,2,7,6),  FaceConnectic(2,3,8,7),  FaceConnectic(3,4,9,8),
  FaceConnectic(4,0,5,9)
};

// Generated by calcul.rb
const SVCFaceConnectic wedge7_svc_face_connectic[15] ARCANE_UNUSED_ATTRIBUTE =
{
 SVCFaceConnectic(6, 9, 1, 5, 9 ),
 SVCFaceConnectic(1, 8, 5, 8, 9 ),
 SVCFaceConnectic(1, 6, 3, 6, 7 ),
 SVCFaceConnectic(4, 13, 5, 3, 8 ),
 SVCFaceConnectic(4, 2, 0, 2, 3 ),
 SVCFaceConnectic(2, 0, 0, 0, 1 ),
 SVCFaceConnectic(1, 7, 4, 7, 8 ),
 SVCFaceConnectic(1, 5, 2, 5, 6 ),
 SVCFaceConnectic(3, 1, 0, 1, 2 ),
 SVCFaceConnectic(6, 10, 2, 0, 5 ),
 SVCFaceConnectic(2, 11, 3, 1, 6 ),
 SVCFaceConnectic(3, 12, 4, 2, 7 ),
 SVCFaceConnectic(0, 4, 6, 0, 4 ),
 SVCFaceConnectic(5, 14, 6, 4, 9 ),
 SVCFaceConnectic(5, 3, 0, 3, 4 )
};

// tableaux de constantes pour les prismes à base hexagonale
//-----------------------------------------------------------

const Integer wedge8_node_association[12]={0,1,2,3,4,5,6,7,8,9,10,11};
const NodeConnectic wedge8_node_connectic[12] ARCANE_UNUSED_ATTRIBUTE =
{
  NodeConnectic( /* Edges */ 0, 5, 12, /* Faces */ 0, 7, 2, /* Nodes */ 0, 1, 5, 6),
  NodeConnectic( /* Edges */ 1, 0, 13, /* Faces */ 0, 2, 3, /* Nodes */ 1, 2, 0, 7),
  NodeConnectic( /* Edges */ 2, 1, 14, /* Faces */ 0, 3, 4, /* Nodes */ 2, 3, 1, 8),
  NodeConnectic( /* Edges */ 3, 2, 15, /* Faces */ 0, 4, 5, /* Nodes */ 3, 4, 2, 9),
  NodeConnectic( /* Edges */ 4, 3, 16, /* Faces */ 0, 5, 6, /* Nodes */ 4, 5, 3, 10),
  NodeConnectic( /* Edges */ 5, 4, 17, /* Faces */ 0, 6, 7, /* Nodes */ 5, 0, 4, 11),
  NodeConnectic( /* Edges */ 11, 6, 12, /* Faces */ 1, 2, 7, /* Nodes */ 6, 11, 7, 0),
  NodeConnectic( /* Edges */ 6, 7, 13, /* Faces */ 1, 3, 2, /* Nodes */ 7, 6, 8 ,1),
  NodeConnectic( /* Edges */ 7, 8, 14, /* Faces */ 1, 4, 3, /* Nodes */ 8, 7, 9, 2),
  NodeConnectic( /* Edges */ 8, 9, 15, /* Faces */ 1, 5, 4, /* Nodes */ 9, 8, 10, 3),
  NodeConnectic( /* Edges */ 9, 10, 16, /* Faces */ 1, 6, 5, /* Nodes */ 10, 9, 11, 4),
  NodeConnectic( /* Edges */ 10, 11, 17, /* Faces */ 1, 7, 6, /* Nodes */ 11, 10, 6, 5)
};

const EdgeConnectic wedge8_edge_connectic[18] ARCANE_UNUSED_ATTRIBUTE =
{
  EdgeConnectic( 0, 1, 0, 2), EdgeConnectic( 1, 2, 0, 3),
  EdgeConnectic( 2, 3, 0, 4), EdgeConnectic( 3, 4, 0, 5),
  EdgeConnectic( 4, 5, 0, 6), EdgeConnectic( 5, 0, 0, 7),
  EdgeConnectic( 6, 7, 2, 1), EdgeConnectic( 7, 8, 3, 1),
  EdgeConnectic( 8, 9, 4, 1), EdgeConnectic( 9,10, 5, 1),
  EdgeConnectic(10,11, 6, 1), EdgeConnectic(11, 6, 7, 1),
  EdgeConnectic( 0, 6, 7, 2), EdgeConnectic( 1, 7, 2, 3),
  EdgeConnectic( 2, 8, 3, 4), EdgeConnectic( 3, 9, 4, 5),
  EdgeConnectic( 4,10, 5, 6), EdgeConnectic( 5,11, 6, 7)

};

const FaceConnectic wedge8_face_connectic[8] ARCANE_UNUSED_ATTRIBUTE =
{
  FaceConnectic( 0, 5, 4, 3, 2, 1), FaceConnectic( 6, 7, 8, 9,10,11),
  FaceConnectic( 0, 1, 7, 6      ), FaceConnectic( 1, 2, 8, 7      ),
  FaceConnectic( 2, 3, 9, 8      ), FaceConnectic( 3, 4,10, 9      ),
  FaceConnectic( 4, 5,11,10      ), FaceConnectic( 5, 0, 6,11      )
};

// Generated by calcul.rb
const SVCFaceConnectic wedge8_svc_face_connectic[18] ARCANE_UNUSED_ATTRIBUTE =
{
 SVCFaceConnectic(1, 9, 5, 9, 10 ),
 SVCFaceConnectic(1, 8, 4, 8, 9 ),
 SVCFaceConnectic(7, 12, 2, 0, 6 ),
 SVCFaceConnectic(7, 11, 1, 6, 11 ),
 SVCFaceConnectic(6, 17, 7, 5, 11 ),
 SVCFaceConnectic(4, 15, 5, 3, 9 ),
 SVCFaceConnectic(2, 13, 3, 1, 7 ),
 SVCFaceConnectic(0, 5, 7, 0, 5 ),
 SVCFaceConnectic(1, 10, 6, 10, 11 ),
 SVCFaceConnectic(1, 6, 2, 6, 7 ),
 SVCFaceConnectic(6, 4, 0, 4, 5 ),
 SVCFaceConnectic(5, 16, 6, 4, 10 ),
 SVCFaceConnectic(4, 2, 0, 2, 3 ),
 SVCFaceConnectic(2, 0, 0, 0, 1 ),
 SVCFaceConnectic(1, 7, 3, 7, 8 ),
 SVCFaceConnectic(3, 1, 0, 1, 2 ),
 SVCFaceConnectic(5, 3, 0, 3, 4 ),
 SVCFaceConnectic(3, 14, 4, 2, 8 )
};

// tableaux de constantes pour les quadrangles
//---------------------------------------------

// noeuds du quadrangle
const Integer quad_node_association[4]={0,1,2,3};

// pour chaque noeud, on donne les 2 faces auxquelles il est connecté
const NodeConnectic quad_node_connectic[4] ARCANE_UNUSED_ATTRIBUTE =
{
   NodeConnectic(  /* Faces */ 0, 3, /* Nodes */ 0, 1, 3),
   NodeConnectic(  /* Faces */ 1, 0, /* Nodes */ 1, 2, 0),
   NodeConnectic(  /* Faces */ 2, 1, /* Nodes */ 2, 3, 1),
   NodeConnectic(  /* Faces */ 3, 2, /* Nodes */ 3, 0, 2),
};

// pour chaque face, on donne les 2 noeuds qui la composent 
const FaceConnectic quad_face_connectic[4] ARCANE_UNUSED_ATTRIBUTE =
{
  FaceConnectic(0,1),FaceConnectic(1,2),FaceConnectic(2,3),FaceConnectic(3,0)
};

// Generated by calcul.rb
const SVCFaceConnectic quad_svc_face_connectic[4] ARCANE_UNUSED_ATTRIBUTE =
{  
  SVCFaceConnectic(0, 0, 1 ),
  SVCFaceConnectic(1, 1, 2 ),
  SVCFaceConnectic(2, 2, 3 ),
  SVCFaceConnectic(3, 3, 0 ),
};
// tableaux de constantes pour les triangles
//------------------------------------------

const Integer triangle_node_association[4]={0,1,2,0};

const NodeConnectic triangle_node_connectic[4] ARCANE_UNUSED_ATTRIBUTE =
{
   NodeConnectic(  /* Faces */ 0, 3, /* Nodes */ 0, 1, 3),
   NodeConnectic(  /* Faces */ 1, 0, /* Nodes */ 1, 2, 0),
   NodeConnectic(  /* Faces */ 2, 1, /* Nodes */ 2, 0, 1),
   NodeConnectic(  /* Faces */ 3, 2, /* Nodes */ 0, 0, 2),
};


const FaceConnectic triangle_face_connectic[4] ARCANE_UNUSED_ATTRIBUTE =
{
  FaceConnectic(0,1),
  FaceConnectic(1,2),
  FaceConnectic(2,0),
  FaceConnectic(0,0),

};

// Generated by calcul.rb
const SVCFaceConnectic triangle_svc_face_connectic[4] ARCANE_UNUSED_ATTRIBUTE =
{  
  SVCFaceConnectic(0, 0, 1 ),
  SVCFaceConnectic(1, 1, 2 ),
  SVCFaceConnectic(2, 2, 0 ),
  SVCFaceConnectic(3, 0, 0 ),
};


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Tableau de 12 éléments de type réel. Utilisé pour la pondération des
 *        des noeuds des mailles (la plus grosse ayant 12 noeuds)
 */
struct LocalCellNodeReal
{
 private:
  Real m_node[12];
 public:
  inline Real node(Integer i) const { return m_node[i]; }
 public:

  LocalCellNodeReal(Real a0,Real a1,Real a2 ,Real a3,
		    Real a4,Real a5,Real a6 ,Real a7,
		    Real a8,Real a9,Real a10,Real a11)
  {
    m_node[0] = a0; m_node[1] = a1; m_node[ 2] = a2;  m_node[ 3] = a3;
    m_node[4] = a4; m_node[5] = a5; m_node[ 6] = a6;  m_node[ 7] = a7;
    m_node[8] = a8; m_node[9] = a9; m_node[10] = a10; m_node[11] = a11;
  }  

 
 public:
  Real sum() const
    { 
       return (m_node[0] + m_node[1] + m_node[2] + m_node[3] + m_node[4]
	      + m_node[5] + m_node[6] + m_node[7]
	      + m_node[8] + m_node[9] + m_node[10] + m_node[11]);
    }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GEOMETRIC_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
