// An independent UCI-facing harness; all play/search delegates to upstream OpenTafl.
import java.io.*;
import java.util.*;
import com.manywords.softworks.tafl.engine.*;
import com.manywords.softworks.tafl.engine.ai.*;
import com.manywords.softworks.tafl.rules.*;
import com.manywords.softworks.tafl.rules.copenhagen.Copenhagen;
import com.manywords.softworks.tafl.notation.PositionSerializer;

public class OpenTaflBridge {
    static PrintStream wire = System.out;
    static Game game;
    static void reset() { game = new Game(Copenhagen.newCopenhagen11(), null); }
    public static void main(String[] args) throws Exception {
        // Redirect upstream diagnostics before any rules/search initialization.
        System.setOut(System.err);
        Coord.initialize();
        reset();
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
        String line;
        while ((line=in.readLine())!=null) {
            try {
                String[] p=line.trim().split("\\s+");
                if(p[0].equals("uci")) { wire.println("id name OpenTafl upstream bridge"); wire.println("uciok"); }
                else if(p[0].equals("isready")) wire.println("readyok");
                else if(p[0].equals("ucinewgame")) reset();
                else if(p[0].equals("position")) {
                    reset();
                    if(!p[1].equals("startpos")) throw new Exception("startpos + moves required");
                    for(int i=3;i<p.length;i++) {
                        String m=p[i].replaceFirst("([a-k][0-9]+)([a-k])", "$1-$2");
                        int result=game.getCurrentState().makeMove(MoveRecord.getMoveRecordFromSimpleString(m));
                        if(result<0) throw new Exception("upstream rejected move "+m+": "+result);
                    }
                }
                else if(p[0].equals("go")) {
                    int millis=Integer.parseInt(p[2]);
                    long started=System.nanoTime();
                    AiWorkspace work=new AiWorkspace(null,game,game.getCurrentState(),32);
                    work.chatty=false;
                    work.allowContinuation(false); work.allowHorizon(false);
                    Thread task=new Thread(()->work.explore(1)); task.start();
                    long setupMillis=(System.nanoTime()-started)/1000000;
                    Thread.sleep(Math.max(1,millis-setupMillis-150)); work.crashStop();
                    long remaining=millis-(System.nanoTime()-started)/1000000-5;
                    task.join(Math.max(1,remaining));
                    if(task.isAlive()) {
                        wire.println("info string error deadline: OpenTafl cancellation exceeded budget"); wire.flush();
                        continue;
                    }
                    GameTreeNode best=work.getTreeRoot().getBestChild();
                    wire.println("bestmove "+(best==null?"0000":best.getEnteringMove().toSimpleString().replace("-","")));
                    wire.flush();
                    work.stopExploring();
                }
                else if(p[0].equals("board")) wire.println("board "+PositionSerializer.getPositionRecord(game.getCurrentState().getBoard()));
                else if(p[0].equals("quit")) break;
                else if(!p[0].equals("stop")) throw new Exception("unsupported command");
            } catch(Exception e) { wire.println("info string error "+e.toString()); }
            wire.flush();
        }
        System.exit(0);
    }
}
