// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <QHyodaGLCell.h>
#include <QOpenGLFunctions>

/******************************************************************************
 * QHyodaGLCell class
 *****************************************************************************/
QHyodaGLCell::QHyodaGLCell(QWidget *parent):QHyodaGL(parent){
  qDebug()<<"\33[36m[QHyodaGLCell::QHyodaGLCell] NEW\33[m";
}


void QHyodaGLCell::clear(){
  nodes.clear();
  colors.clear();
}

void QHyodaGLCell::add_node(const QVector3D &node_coords, const QColor color){
  nodes.append(node_coords);
  colors.append(QVector4D(color.redF(),color.greenF(),color.blueF(),color.alphaF()));
}


/******************************************************************************
 * draw
 *****************************************************************************/
void QHyodaGLCell::draw() const{
  //qDebug()<<"\t\33[36m[QHyodaGLCell:draw]\33[m";
  const QColor color=QColor(Qt::green);

  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_COLOR_ARRAY);

  // Loading vertices & colors
  glVertexPointer(3, GL_FLOAT, 0, nodes.constData());
  glColorPointer(4, GL_FLOAT, 0, colors.constData());
   
  glPushMatrix();

  // On fait les pointillÃ©s derriÃ¨re
  glLineWidth(1.0);
  glEnable(GL_LINE_STIPPLE);  
  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  if (cell_nb_nodes==3) glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_BYTE, stdIdx);
  if (cell_nb_nodes==4) glDrawElements(GL_QUADS, 4, GL_UNSIGNED_BYTE, stdIdx);
  if (cell_nb_nodes==8) glDrawElements(GL_QUADS, 24, GL_UNSIGNED_BYTE, hexIdx);
  
  // Et ou revient pour les arÃªtes du devant
  glLineWidth(2.0);
  glDisable(GL_LINE_STIPPLE);
  glPolygonMode(GL_FRONT, GL_LINE);
  glPolygonMode(GL_BACK, GL_POINT);
  if (cell_nb_nodes==3) glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_BYTE, stdIdx);
  if (cell_nb_nodes==4) glDrawElements(GL_QUADS, 4, GL_UNSIGNED_BYTE, stdIdx);
  if (cell_nb_nodes==8) glDrawElements(GL_QUADS, 24, GL_UNSIGNED_BYTE, hexIdx);
  
  // Les noeuds
  glDisableClientState(GL_COLOR_ARRAY);
  glColor4f(color.redF(), color.greenF(), color.blueF(), color.alphaF());
  glDrawElements(GL_POINTS, 8, GL_UNSIGNED_BYTE, stdIdx);
  glEnableClientState(GL_COLOR_ARRAY);
  
  glPopMatrix();
  
  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_COLOR_ARRAY);
}
