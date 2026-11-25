// BRACS の QuPath 注釈（.qpdata）を GeoJSON に書き出す。画像は開かない。
// 使い方: QuPath script -a <入力ディレクトリ> -a <出力ディレクトリ> export_qpdata_geojson.groovy
import qupath.lib.io.PathIO

def inDir = new File(args[0])
def outDir = new File(args[1])
int nFiles = 0
int nObjects = 0
int nFailed = 0
inDir.eachFileRecurse { f ->
    if (!f.name.endsWith('.qpdata')) return
    def rel = inDir.toPath().relativize(f.toPath()).toString().replace('.qpdata', '.geojson')
    def out = new File(outDir, rel)
    out.parentFile.mkdirs()
    try {
        def hierarchy = PathIO.readHierarchy(f)
        def objects = hierarchy.getAnnotationObjects()
        PathIO.exportObjectsAsGeoJSON(out, objects, PathIO.GeoJsonExportOptions.FEATURE_COLLECTION)
        nFiles++
        nObjects += objects.size()
    } catch (Exception e) {
        nFailed++
        println "FAILED ${f}: ${e}"
    }
}
println "exported files=${nFiles} annotations=${nObjects} failed=${nFailed}"
if (nFailed > 0) System.exit(1)
