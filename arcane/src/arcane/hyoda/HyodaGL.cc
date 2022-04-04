// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*****************************************************************************
 * HyodaGL.cc from GLUT                                        (C) 2000-2012 *
 *****************************************************************************/
//#include "arcane/IMesh.h"
#include "arcane/IApplication.h"
#include "arcane/IParallelMng.h"
#include "arcane/FactoryService.h"
#include "arcane/ServiceFinder2.h"
#include "arcane/SharedVariable.h"
#include "arcane/CommonVariables.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/AbstractService.h"
#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/AbstractService.h"
#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/hyoda/HyodaIceT.h"

#include <IceTConfig.h>
#include <IceT.h>
#include <IceTGL.h>

#include <GL/osmesa.h>
#include "GL/glu.h"


void ppmWrite(const char *filename,
              const unsigned char *image,
              int width, int height){
  FILE *fd;
  const unsigned char *color;
  fd = fopen(filename, "wb");
  fprintf(fd, "P6\n");
  fprintf(fd, "%d %d\n", width, height);
  fprintf(fd, "255\n");
  for(int y = height-1; y >= 0; y--) {
    color = image + y*width*4;
    for(int x = 0; x < width; x++) {
      fwrite(color, 1, 3, fd);
      color += 4;
    }
  }
  fclose(fd);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


void HyodaIceT::gluSphere(float radius, int slices, int stacks){
   GLUquadric *q = gluNewQuadric();
   gluQuadricNormals(q, GLU_SMOOTH);
   ::gluSphere(q, radius, slices, stacks);
   gluDeleteQuadric(q);
}


void HyodaIceT::gluCone(float base, float height, int slices, int stacks){
   GLUquadric *q = gluNewQuadric();
   gluQuadricDrawStyle(q, GLU_FILL);
   gluQuadricNormals(q, GLU_SMOOTH);
   gluCylinder(q, base, 0.0, height, slices, stacks);
   gluDeleteQuadric(q);
}


void HyodaIceT::gluTorus(float innerRadius, float outerRadius, int sides, int rings){
   int i, j;
   GLfloat theta, phi, theta1;
   GLfloat cosTheta, sinTheta;
   GLfloat cosTheta1, sinTheta1;
   const GLfloat ringDelta = 2.0 * M_PI / rings;
   const GLfloat sideDelta = 2.0 * M_PI / sides;

   theta = 0.0;
   cosTheta = 1.0;
   sinTheta = 0.0;
   for (i = rings - 1; i >= 0; i--) {
      theta1 = theta + ringDelta;
      cosTheta1 = cos(theta1);
      sinTheta1 = sin(theta1);
      glBegin(GL_QUAD_STRIP);
      phi = 0.0;
      for (j = sides; j >= 0; j--) {
         GLfloat cosPhi, sinPhi, dist;

         phi += sideDelta;
         cosPhi = cos(phi);
         sinPhi = sin(phi);
         dist = outerRadius + innerRadius * cosPhi;

         glNormal3f(cosTheta1 * cosPhi, -sinTheta1 * cosPhi, sinPhi);
         glVertex3f(cosTheta1 * dist, -sinTheta1 * dist, innerRadius * sinPhi);
         glNormal3f(cosTheta * cosPhi, -sinTheta * cosPhi, sinPhi);
         glVertex3f(cosTheta * dist, -sinTheta * dist,  innerRadius * sinPhi);
      }
      glEnd();
      theta = theta1;
      cosTheta = cosTheta1;
      sinTheta = sinTheta1;
   }
}


void HyodaIceT::renderSphereConeTorus(void){
  GLfloat red_mat[]   = { 1.0, 0.2, 0.2, 1.0 };
  GLfloat green_mat[] = { 0.2, 1.0, 0.2, 1.0 };
  GLfloat blue_mat[]  = { 0.2, 0.2, 1.0, 1.0 };

  glRotatef(20.0, 1.0, 0.0, 0.0);

  glPushMatrix();
  glTranslatef(-0.75, 0.5, 0.0); 
  glRotatef(90.0, 1.0, 0.0, 0.0);
  glMaterialfv( GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, red_mat );
  gluTorus(0.275, 0.85, 20, 20);
  glPopMatrix();

  glPushMatrix();
  glTranslatef(-0.75, -0.5, 0.0); 
  glRotatef(270.0, 1.0, 0.0, 0.0);
  glMaterialfv( GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, green_mat );
  gluCone(1.0, 2.0, 16, 1);
  glPopMatrix();

  glPushMatrix();
  glTranslatef(0.75, 0.0, -1.0); 
  glMaterialfv( GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, blue_mat );
  gluSphere(1.0, 20, 20);
  glPopMatrix();

  /* This is very important!!!
   * Make sure buffered commands are finished!!!
   */
  glFinish();
}


/* Encode image position in color. */
#define ACTIVE_COLOR(x, y) ((((x) & 0xFFFF) | 0x8000) | ((((y) & 0xFFFF) | 0x8000) << 16))
void HyodaIceT::LowerTriangleImage(void *img){
  IceTImage image=*(IceTImage*)(&img);
  //IceTUInt *data = icetImageGetColorui(image);
  IceTUByte *data = icetImageGetColorub(image);
  IceTSizeType width = icetImageGetWidth(image);
  IceTSizeType height = icetImageGetHeight(image);
  IceTSizeType x, y;

  for (y = 0; y < height; y++) {
    for (x = 0; x < width; x++) {
      if (x < (height-y)) {
        data[0] = ACTIVE_COLOR(x, y);
      }
      data++;
    }
  }
}


void HyodaIceT::UpperTriangleImage(void *img){
  IceTImage image=*(IceTImage*)(&img);
  //IceTUInt *data = icetImageGetColorui(image);
  IceTUByte *data = icetImageGetColorub(image);
  IceTSizeType width = icetImageGetWidth(image);
  IceTSizeType height = icetImageGetHeight(image);
  IceTSizeType x, y;

  for (y = 0; y < height; y++) {
    for (x = 0; x < width; x++) {
      if ((height-y) < x) {
        data[0] = ACTIVE_COLOR(x, y);
      }
      data++;
    }
  }
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

