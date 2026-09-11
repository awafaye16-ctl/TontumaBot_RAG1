from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('node'), 'Node required for browser script checks')
class FrontendTests(unittest.TestCase):
    def run_js(self, source):
        result = subprocess.run(['node', '-'], input=source, capture_output=True, text=True, encoding='utf-8', timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_recording_mime_matches_extension(self):
        html = (ROOT / 'static/index.html').read_text(encoding='utf-8')
        script = html[html.index('    function recordingExtension'):html.index('    function fmtTime')]
        self.run_js(script + "const assert=require('node:assert/strict'); assert.equal(recordingExtension('audio/mp4;codecs=mp4a.40.2'), '.m4a'); assert.equal(recordingExtension('audio/ogg; codecs=opus'), '.ogg'); assert.equal(recordingExtension('audio/webm'), '.webm');")

    def test_admin_auth_header_and_clear_error(self):
        html = (ROOT / 'static/admin.html').read_text(encoding='utf-8')
        helper = html[html.index('        function adminFetch'):html.index('        async function refresh')]
        self.run_js('''
            const assert = require('node:assert/strict');
            const $ = id => id === 'admin-key' ? {value:'fixture-key'} : {addEventListener(){}};
            const fetch = async (path, options) => {
                assert.equal(options.headers.get('Authorization'), 'Bearer fixture-key');
                assert.equal(path, '/admin/documents');
            };
        ''' + helper + "adminFetch('/admin/documents').catch(e=>{console.error(e);process.exitCode=1});")
        handler = html[html.index("        $('btn-clear-all').addEventListener"):html.index('        /* ── Init')]
        self.run_js('''
            const assert = require('node:assert/strict');
            let click, message = '', refreshes = 0;
            const $ = () => ({addEventListener: (type, callback) => {click=callback}});
            const confirm = () => true;
            const adminFetch = async () => ({ok:false, json:async()=>({detail:'Authentication required'})});
            const toast = value => {message=value};
            const refresh = () => {refreshes++};
        ''' + handler + '''
            click().then(()=>{assert.equal(message,'Authentication required');assert.equal(refreshes,0)}).catch(e=>{console.error(e);process.exitCode=1});
        ''')

    def test_inline_scripts_parse(self):
        import json
        for page in ('index.html', 'admin.html'):
            html = (ROOT / 'static' / page).read_text(encoding='utf-8')
            script = html.split('<script>')[-1].split('</script>')[0]
            self.run_js('new (require("node:vm").Script)(' + json.dumps(script) + ');')

    def test_stream_disconnect_unlocks_and_partial_error_never_retries(self):
        html = (ROOT / 'static/index.html').read_text(encoding='utf-8')
        script = html[html.index('    let wsAsk = null;'):html.index('    async function sendTextHTTP')]
        self.run_js('''
            const assert = require('node:assert/strict');
            let sockets = [], fallbacks = 0;
            const controls = {send: {disabled:false}, input: {value:''}, 'tts-engine':{value:''}, language:{value:''}};
            const $ = id => controls[id];
            const location = {protocol:'http:', host:'localhost'};
            class WebSocket { constructor(){sockets.push(this)} close(){if(this.onclose)this.onclose()} send(){} }
            const textNode = {textContent:''};
            const cursor = {remove(){}};
            const bubble = {querySelector: id => id === '.cursor' ? cursor : textNode};
            const addMsg = () => ({}), getBubble = () => bubble;
            const addTyping = () => {}, removeTyping = () => {}, autoGrow = () => {}, scrollBot = () => {};
            const esc = value => value;
            const toast = () => {};
            const sendTextHTTP = async () => {fallbacks++};
        ''' + script + '''
            (async () => {
                await sendText('Question', false);
                sockets.at(-1).onclose();
                assert.equal(controls.send.disabled, false, 'Send must unlock after interrupted socket');
                await sendText('Question', false);
                const ws = sockets.at(-1);
                ws.onmessage({data:JSON.stringify({type:'lang', lang:'fr'})});
                ws.onmessage({data:JSON.stringify({type:'token', text:'Unvalidated answer'})});
                ws.onerror();
                assert.equal(fallbacks, 0, 'Do not send duplicate requests after processing started');
                assert.notEqual(textNode.textContent, 'Unvalidated answer');
                assert.equal(controls.send.disabled, false);
            })().catch(e => {console.error(e); process.exitCode=1});
        ''')


if __name__ == '__main__':
    unittest.main()
