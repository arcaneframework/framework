using System;
using Arcane;
using Real = System.Double;

#if ARCANE_64BIT
using Integer = System.Int64;
using IntegerConstArrayView = Arcane.Int64ConstArrayView;
using IntegerArrayView = Arcane.Int64ArrayView;
#else
using Integer = System.Int32;
using IntegerConstArrayView = Arcane.Int32ConstArrayView;
using IntegerArrayView = Arcane.Int32ArrayView;
#endif

using System.Collections.Generic;

[Arcane.Service("DotNetDataWriter",typeof(Arcane.IDataWriter))]
public class DotNetDataWriter : Arcane.IDataWriter_WrapperService
{
  public DotNetDataWriter(ServiceBuildInfo bi) : base(bi)
  {
  }

  public override void BeginWrite(VariableCollection vars) {}
  public override void EndWrite() {}

  public override void SetMetaData(string meta_data) {}
  /*!
   * Cette methode sert uniquement pour tester le multi-threading
   * dans le wrapping C#. Elle est appelee par TaskUnitTestCS dans
   * une boucle multi-thread.
   * Le but est de faire des allocations/desallocations pour
   * tester l'utilisation du GarbageCollector dans ce cas.
   */
  public override void Write(IVariable var,IData data)  
  {
    Console.WriteLine("C# WRITE!!! var={0}",var.Name());
    int total = 0;
    int size = 0;
    for(int z=0; z<10; ++z){
      ISerializedData_Ref sdata = data.CreateSerializedDataRef(true);
      ByteConstArrayView bytes = sdata.Get().Buffer();
      byte[] b = bytes.ToArray();
      size = b.Length;
      int n = 2000000;
      int[] x = new int[n];
      for( int i=0; i<n; ++i ){
        int mul = (i>=size) ? n : b[i];
        x[i] = mul*size;
      }
      total += x[0];
      if ((z%3)==0)
        GC.Collect();
    }
    Console.WriteLine("C# WRITE LAST_N={0} S={1}",total,size);
  }
}
