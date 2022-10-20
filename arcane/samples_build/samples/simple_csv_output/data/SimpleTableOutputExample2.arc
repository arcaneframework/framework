<?xml version="1.0"?>
<case codename="csv" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Examples CSV</title>
    <timeloop>example2</timeloop>
  </arcane>

  <meshes>
    <mesh>
      <generator name="Cartesian2D" >
        <face-numbering-version>1</face-numbering-version>

        <nb-part-x>1</nb-part-x> 
        <nb-part-y>1</nb-part-y>

        <origin>0.0 0.0</origin>

        <x>
          <length>1.0</length>
          <n>1</n>
        </x>

        <y>
          <length>1.0</length>
          <n>1</n>
        </y>

      </generator>
    </mesh>
  </meshes>

  <!-- //! [SimpleTableOutputExample2_arc]  -->
  <simple-table-output-example2>

    <!-- Le nom du répertoire à créer/utiliser. -->
    <tableDir>example2</tableDir>
    
    <!-- Le nom du fichier à créer/écraser. -->
    <tableName>Results_Example2</tableName>

    <!-- Au final, on aura un fichier ayant comme chemin : ./example2/Results_Example2.csv -->
  </simple-table-output-example2>
  <!-- //! [SimpleTableOutputExample2_arc]  -->

</case>
