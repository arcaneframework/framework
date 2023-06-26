using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;

namespace Arcane.Axl.Xsd
{
  public partial class Complex
  {
    private bool extended_computed_ = false;
    private List<ExtendedType> extended_ = new List<ExtendedType> ();

    private bool complex_computed_ = false;
    private List<Complex> complex_ = new List<Complex>();

    private bool xml_computed_ = false;
    private OptionHandler xml_;

    public IEnumerable<ExtendedType> Extended { get { return LazyExtended (); } }
    
    public IEnumerable<Complex> FlatteningComplex { get { return LazyComplex(); } }

    public bool IsRef { get { return @ref != null; } }

    public bool HasInterface { get { return @interface != null; } }

    public OptionHandler Xml { get { return LazyXml (); } }

    public Complex ()
    {
    }

    private OptionHandler LazyXml()
    {
      if (xml_computed_)
        return xml_;

      xml_computed_ = true;

      xml_ = new OptionHandler (this);

      return xml_;
    }

    private IEnumerable<ExtendedType> LazyExtended()
    {
        if (extended_computed_)
        return extended_;

        extended_computed_ = true;

      if(@ref != null)
        return extended_;
      if (Items != null) {
        foreach (var item in Items) {
          if (item == null)
            continue;
          if (item is ExtendedType)
            extended_.Add (item as ExtendedType);
          if (item is Complex)
            extended_.AddRange ((item as Complex).Extended);
        }
      }
      return extended_;
    }

    private IEnumerable<Complex> LazyComplex()
    {
        if (complex_computed_)
            return complex_;

        extended_computed_ = true;

        if (@ref != null)
            return complex_;
        if (Items != null) {
          foreach (var item in Items.Where(p => p is Complex)) {
            var complex = item as Complex;
            complex_.AddRange (complex.FlatteningComplex);    
            complex_.Add (complex);
          }
        }
        return complex_;
    }
  }
}

