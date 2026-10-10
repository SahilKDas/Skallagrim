"""Build pinned OpenTafl with modern javac, avoiding its obsolete Gradle wrapper."""

import hashlib
import subprocess
import urllib.request
from transport import ROOT
from records import save


def main():
    repo = ROOT / "external/opentafl"
    jars = repo / "lib"
    jars.mkdir(exist_ok=True)
    dependencies = {
        "ini4j-0.5.4.jar": "org/ini4j/ini4j/0.5.4/ini4j-0.5.4.jar",
        "jline-2.14.1.jar": "jline/jline/2.14.1/jline-2.14.1.jar",
        "javax.json-1.0.4.jar": "org/glassfish/javax.json/1.0.4/javax.json-1.0.4.jar",
    }
    for filename, url in dependencies.items():
        path = jars / filename
        if not path.exists():
            urllib.request.urlretrieve(
                "https://repo.maven.apache.org/maven2/" + url, path
            )
    classes = repo / "build/classes/java/main"
    classes.mkdir(parents=True, exist_ok=True)
    # Compatibility copies: legacy imports plus cancellation memory visibility.
    copies = []
    for source in (repo / "src").rglob("*.java"):
        text = source.read_text(encoding="utf8")
        text = text.replace("import sun.net.NetworkClient;", "")
        text = text.replace(
            "public boolean mNoTime = false;",
            "public volatile boolean mNoTime = false;",
        )
        text = text.replace(
            "import com.sun.org.apache.xerces.internal.impl.dv.util.Base64;",
            "import java.util.Base64;",
        )
        text = text.replace(
            "Base64.encode(salt)", "Base64.getEncoder().encodeToString(salt)"
        ).replace("Base64.encode(bytes)", "Base64.getEncoder().encodeToString(bytes)")
        copy = repo / "build/compat-src" / source.relative_to(repo / "src")
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_text(text, encoding="utf8")
        copies.append(copy)
    sources = repo / "javac-sources.txt"
    sources.write_text("\n".join('"' + str(p).replace("\\", "/") + '"' for p in copies))
    subprocess.run(
        [
            "javac",
            "--release",
            "8",
            "-encoding",
            "UTF-8",
            "-cp",
            str(jars / "*"),
            "-d",
            str(classes),
            "@" + str(sources),
        ],
        check=True,
    )
    bridge = repo / "bridge"
    bridge.mkdir(exist_ok=True)
    subprocess.run(
        [
            "javac",
            "-cp",
            str(classes) + __import__("os").pathsep + str(jars / "*"),
            "-d",
            str(bridge),
            str(ROOT / "python/adapters/OpenTaflBridge.java"),
        ],
        check=True,
    )
    save(
        ROOT / "benchmarks/opentafl-build.json",
        dict(
            jars={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in jars.glob("*.jar")
            },
            clock_patch="AiWorkspace.mNoTime made volatile for cross-thread cancellation; search/rules decisions unchanged",
            bridge_sha256=hashlib.sha256(
                (ROOT / "python/adapters/OpenTaflBridge.java").read_bytes()
            ).hexdigest(),
        ),
    )


if __name__ == "__main__":
    main()
