%module(directors="1") ArcaneHdf5

%import core/ArcaneSwigCore.i

// A partir de HDF5 1.10, hid_t est de type 'Int64'.
// Avant, il est de type 'int'.
// Comme ce paramêtre n'est pas utilisé par référence, on
// utilise toujours un Int64.
typedef Int64 hid_t;

%{
#include "arcane/std/Hdf5Utils.h"
#include "ArcaneSwigUtils.h"
using namespace Arcane;
// Vérifie que 'hid_t' est bien de type 'Int64'
static_assert(sizeof(hid_t) == 8, "bad size of 'hid_t'. must be 8 bytes. Your need HDF5 version 1.10+. Dis version of HDF5 is ");
%}

ARCANE_STD_EXHANDLER
%include hdf5/Hdf5Utils.i
%exception;
