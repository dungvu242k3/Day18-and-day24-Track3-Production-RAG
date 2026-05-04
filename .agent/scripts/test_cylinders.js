const { createClient } = require('@supabase/supabase-js');
require('dotenv').config({path: '.env'});
const supabase = createClient(process.env.VITE_SUPABASE_URL, process.env.VITE_SUPABASE_ANON_KEY);
async function main() {
    const {data, error} = await supabase.from('cylinders').select('id, serial_number, volume, gas_type, valve_type, category, handle_type, customer_name, customer_id').limit(5);
    console.log(JSON.stringify(data, null, 2));
    if(error) console.error(error);
}
main();
